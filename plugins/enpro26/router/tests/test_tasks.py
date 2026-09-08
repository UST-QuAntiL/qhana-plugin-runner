# Copyright 2026 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import threading
from types import SimpleNamespace

import pytest
from flask import current_app
from requests.exceptions import Timeout

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile
from router import tasks as router_tasks
from router import tasks_pipeline_steps as pipeline_steps
from router.schemas import (
    AGGREGATOR_PLUGIN,
    FINALIZE_PIPELINE,
    MAPPING_PLUGIN,
    MDS_PLUGIN,
    NONE_PLUGIN,
    ONE_HOT_PLUGIN,
    PCA_PLUGIN,
    TRANSFORMERS_PLUGIN,
    VECTOR_CONCAT_PLUGIN,
    WU_PALMER_PLUGIN,
)
from router.tasks import handle_webhook_task, preprocessing_task, start_routing_task
from router.tasks_helpers import CELERY_COUNTDOWN
from router.tasks_pipeline_steps import start_wu_palmer
from router.tests.data import (
    PluginServer,
    capture_task_errors,
    input_file_responses,
    make_router_task,
    mock_open_url,
)

from tests.utils import run_task

pytestmark = pytest.mark.usefixtures("celery_worker")

# The tasks the router dispatches, listed per module that looks them up.
DISPATCHED_TASKS = (
    (
        pipeline_steps,
        ("start_wu_palmer", "start_mapping", "start_vector_concat", "start_pca"),
    ),
    (
        router_tasks,
        (
            "start_transformers",
            "start_aggregator",
            "start_mds",
            "finalize_pipeline",
            "finalize_vector_concat",
            "finalize_pca",
        ),
    ),
)


@pytest.fixture
def server(monkeypatch) -> PluginServer:
    return PluginServer().install(monkeypatch)


@pytest.fixture
def task_errors(monkeypatch) -> list:
    return capture_task_errors(monkeypatch)


@pytest.fixture
def inline(monkeypatch):
    """Run every dispatched task immediately instead of waiting out its countdown.

    Returns the results and errors the router reports through the plugin runner
    tasks, which are stubbed out here because they need a parent task to run in.
    """

    def run_now(task):
        def apply_async(args=None, kwargs=None, **options):
            # throw=True surfaces a failing step instead of burying it in a
            # result that nobody looks at
            return task.apply(args=args or (), kwargs=kwargs or {}, throw=True)

        return apply_async

    for module, task_names in DISPATCHED_TASKS:
        for name in task_names:
            task = getattr(module, name)
            monkeypatch.setattr(module, name, SimpleNamespace(apply_async=run_now(task)))

    recorded = SimpleNamespace(results=[], errors=[])
    monkeypatch.setattr(
        "router.tasks_pipeline_steps.save_task_result",
        SimpleNamespace(delay=lambda message, db_id: recorded.results.append(message)),
    )
    monkeypatch.setattr(
        "router.tasks_helpers.save_task_error",
        SimpleNamespace(delay=lambda **kwargs: recorded.errors.append(kwargs)),
    )
    return recorded


def run_inline(task, *args):
    return task.apply(args=list(args)).get()


def reload(db_task: ProcessingTask) -> ProcessingTask:
    DB.session.expire_all()
    return ProcessingTask.get_by_id(db_task.id)


def stored_file_names(db_task: ProcessingTask) -> set:
    DB.session.expire_all()
    return {file.file_name for file in TaskFile.get_task_result_files(db_task.id)}


def complete_sub_task(db_task: ProcessingTask, plugin: str, via: str = "webhook"):
    """Deliver the completion event the sub-plugin would send to the webhook."""
    source_url = reload(db_task).data[f"{plugin}_url"]
    return run_inline(handle_webhook_task, db_task.id, source_url, via)


# --- PREPROCESSING ---


def test_preprocessing_finds_the_taxonomy_attributes(monkeypatch):
    mock_open_url(monkeypatch, input_file_responses())
    db_task = make_router_task()

    result = run_task(preprocessing_task, db_id=db_task.id)

    assert result == "Found 2 taxonomy attribute(s)."
    assert set(reload(db_task).data["taxonomy_attributes"]) == {
        "genre",
        "instrumentation",
    }


def test_preprocessing_recommends_a_pipeline_per_attribute(monkeypatch):
    """Only taxonomies that carry mapping data can be routed to the mapping plugin."""
    mock_open_url(monkeypatch, input_file_responses())
    db_task = make_router_task()

    run_task(preprocessing_task, db_id=db_task.id)

    assert reload(db_task).data["recommendations"] == {
        "genre": MAPPING_PLUGIN,
        "instrumentation": WU_PALMER_PLUGIN,
    }


def test_preprocessing_skips_attributes_without_a_usable_taxonomy(monkeypatch):
    """``composer`` refers to a plain file and ``missing_tax`` is not in the zip."""
    mock_open_url(monkeypatch, input_file_responses())
    db_task = make_router_task()

    run_task(preprocessing_task, db_id=db_task.id)

    attributes = reload(db_task).data["taxonomy_attributes"]
    assert "composer" not in attributes
    assert "missing_tax" not in attributes
    assert "not_in_entities" not in attributes


# --- ROUTING ---


@pytest.mark.parametrize(
    "selections,overrides,expected_pipelines,expected_target",
    [
        ({"a": WU_PALMER_PLUGIN}, {}, [WU_PALMER_PLUGIN], 5),
        ({"a": MAPPING_PLUGIN}, {}, [MAPPING_PLUGIN], 4),
        (
            {"a": WU_PALMER_PLUGIN, "b": MAPPING_PLUGIN},
            {},
            [WU_PALMER_PLUGIN, MAPPING_PLUGIN],
            8,
        ),
        ({"a": WU_PALMER_PLUGIN}, {"concatOutput": True}, [WU_PALMER_PLUGIN], 6),
        (
            {"a": WU_PALMER_PLUGIN},
            {"concatOutput": True, "reduceDimensions": True},
            [WU_PALMER_PLUGIN],
            7,
        ),
    ],
)
def test_routing_task_counts_every_plugin_it_will_run(
    monkeypatch, selections, overrides, expected_pipelines, expected_target
):
    launched = []
    for name in ("start_wu_palmer", "start_mapping"):
        monkeypatch.setattr(
            f"router.tasks_pipeline_steps.{name}.apply_async",
            lambda *args, name=name, **kwargs: launched.append(name),
        )
    db_task = make_router_task(selections=selections, **overrides)

    result = run_task(start_routing_task, db_id=db_task.id)

    assert result == "Routing task started and pipeline queued."
    db_task = reload(db_task)
    # the first pipeline of the queue is started right away
    assert [
        db_task.data["current_pipeline"],
        *db_task.data["pipeline_queue"],
    ] == expected_pipelines
    assert db_task.progress_target == expected_target
    assert db_task.progress_value == 1
    assert db_task.progress_unit == "Steps"
    assert launched == [f"start_{expected_pipelines[0]}"]


def test_routing_task_groups_the_attributes_per_pipeline(monkeypatch):
    monkeypatch.setattr(
        "router.tasks_pipeline_steps.start_wu_palmer.apply_async",
        lambda *args, **kwargs: None,
    )
    db_task = make_router_task(
        selections={
            "g1": WU_PALMER_PLUGIN,
            "m1": MAPPING_PLUGIN,
            "g2": WU_PALMER_PLUGIN,
            "skipped": NONE_PLUGIN,
        }
    )

    run_task(start_routing_task, db_id=db_task.id)

    data = reload(db_task).data
    assert data[f"{WU_PALMER_PLUGIN}_attributes"] == "g1\ng2"
    assert data[f"{MAPPING_PLUGIN}_attributes"] == "m1"


def test_routing_task_reports_unsupported_and_skipped_attributes(monkeypatch):
    monkeypatch.setattr(
        "router.tasks_pipeline_steps.start_wu_palmer.apply_async",
        lambda *args, **kwargs: None,
    )
    db_task = make_router_task(
        selections={"a": WU_PALMER_PLUGIN, "hot": ONE_HOT_PLUGIN, "skip": NONE_PLUGIN}
    )

    run_task(start_routing_task, db_id=db_task.id)

    task_log = reload(db_task).task_log
    assert "One-Hot encoding not yet supported" in task_log
    assert "None selected attributes skipped: ['skip']" in task_log


# --- WEBHOOK STATE MACHINE ---


@pytest.mark.parametrize(
    "source_url",
    [
        pytest.param("", id="no source"),
        pytest.param("http://localhost:5005/plugins/other/1/", id="foreign plugin"),
        pytest.param(
            None, id="sub-task of another pipeline"
        ),  # None value is evaluated at runtime in test
    ],
)
def test_webhook_ignores_unknown_sources(server, source_url):
    """The webhook endpoint is unauthenticated, so a bogus event must not fail the task."""
    db_task = make_router_task()
    if source_url is None:
        # a url of a step that is not awaited right now
        source_url = server.task_url(MDS_PLUGIN)

    result = run_task(
        handle_webhook_task, db_id=db_task.id, source_url=source_url, via="webhook"
    )

    assert result == "Unrecognized webhook source"
    db_task = reload(db_task)
    assert db_task.task_status != "FAILURE"
    assert db_task.progress_value == 1


def test_webhook_ignores_a_step_that_is_not_awaited(server):
    """A late watchdog poll for an already finished step must not progress anything."""
    db_task = make_router_task()

    result = run_task(
        handle_webhook_task,
        db_id=db_task.id,
        source_url=server.task_url(MDS_PLUGIN),
        via="watchdog",
    )

    assert result == "Unrecognized webhook source"
    assert reload(db_task).progress_value == 1


def test_webhook_ignores_an_unknown_task(server):
    result = run_task(
        handle_webhook_task,
        db_id=999999,
        source_url="http://localhost:5005/plugins/wu-palmer/tasks/1/",
        via="webhook",
    )

    assert result == "Unknown task"


def test_webhook_waits_for_a_pending_sub_task(server):
    url = server.task_url(WU_PALMER_PLUGIN)
    server.statuses[url] = "PENDING"
    db_task = make_router_task(data={f"{WU_PALMER_PLUGIN}_url": url})

    result = run_task(
        handle_webhook_task, db_id=db_task.id, source_url=url, via="webhook"
    )

    assert result == "Sub-task still pending"


def test_webhook_fails_the_task_when_a_sub_task_failed(server, task_errors):
    url = server.task_url(WU_PALMER_PLUGIN)
    server.statuses[url] = "FAILURE"
    db_task = make_router_task(data={f"{WU_PALMER_PLUGIN}_url": url})

    with pytest.raises(RuntimeError, match="Sub-plugin step failed"):
        run_task(handle_webhook_task, db_id=db_task.id, source_url=url, via="webhook")

    assert [error["db_id"] for error in task_errors] == [db_task.id]


@pytest.mark.parametrize(
    "via,expected_delivery,seen",
    [
        ("webhook", "webhook", True),
        (None, "webhook", True),
        ("watchdog", "watchdog", False),
    ],
)
def test_webhook_records_how_the_event_arrived(
    server, inline, via, expected_delivery, seen
):
    """Only a real webhook delivery proves the event arrived at all."""
    url = server.task_url(WU_PALMER_PLUGIN)
    db_task = make_router_task(data={f"{WU_PALMER_PLUGIN}_url": url})

    run_inline(handle_webhook_task, db_task.id, url, via)

    data = reload(db_task).data
    assert (url in data.get("webhook_seen", {})) is seen
    assert data["progressed_via"] == {url: expected_delivery}


@pytest.mark.parametrize("via", ["webhook", "watchdog"])
def test_webhook_advances_the_progress(server, inline, via):
    url = server.task_url(WU_PALMER_PLUGIN)
    db_task = make_router_task(data={f"{WU_PALMER_PLUGIN}_url": url})

    run_inline(handle_webhook_task, db_task.id, url, via)

    assert reload(db_task).progress_value == 2


def test_webhook_progresses_a_sub_task_only_once(server, inline):
    """The watchdog and the webhook may report the same completion event."""
    url = server.task_url(WU_PALMER_PLUGIN)
    db_task = make_router_task(data={f"{WU_PALMER_PLUGIN}_url": url})
    run_inline(handle_webhook_task, db_task.id, url, "webhook")

    result = run_inline(handle_webhook_task, db_task.id, url, "watchdog")

    assert result == "Sub-task already progressed"
    assert reload(db_task).progress_value == 2


def test_watchdog_reports_a_lost_webhook_event(server, inline):
    url = server.task_url(WU_PALMER_PLUGIN)
    db_task = make_router_task(data={f"{WU_PALMER_PLUGIN}_url": url})

    run_inline(handle_webhook_task, db_task.id, url, "watchdog")

    assert "recovered the lost completion event" in reload(db_task).task_log


def test_watchdog_reports_overtaking_a_slow_webhook_handler(server, inline):
    url = server.task_url(WU_PALMER_PLUGIN)
    db_task = make_router_task(
        data={
            f"{WU_PALMER_PLUGIN}_url": url,
            "webhook_seen": {url: "2026-01-01T00:00:00+00:00"},
        }
    )

    run_inline(handle_webhook_task, db_task.id, url, "watchdog")

    assert "but its handler was still retrying" in reload(db_task).task_log


@pytest.mark.parametrize(
    "pipeline,source_plugin,next_task",
    [
        (WU_PALMER_PLUGIN, WU_PALMER_PLUGIN, "start_transformers"),
        (WU_PALMER_PLUGIN, TRANSFORMERS_PLUGIN, "start_aggregator"),
        (WU_PALMER_PLUGIN, AGGREGATOR_PLUGIN, "start_mds"),
        (WU_PALMER_PLUGIN, MDS_PLUGIN, "finalize_pipeline"),
        (MAPPING_PLUGIN, MAPPING_PLUGIN, "start_aggregator"),
        (MAPPING_PLUGIN, AGGREGATOR_PLUGIN, "start_mds"),
        (MAPPING_PLUGIN, MDS_PLUGIN, "finalize_pipeline"),
        (FINALIZE_PIPELINE, VECTOR_CONCAT_PLUGIN, "finalize_vector_concat"),
        (FINALIZE_PIPELINE, PCA_PLUGIN, "finalize_pca"),
    ],
)
def test_webhook_triggers_the_next_step_of_the_pipeline(
    monkeypatch, server, pipeline, source_plugin, next_task
):
    url = server.task_url(source_plugin)
    db_task = make_router_task(
        data={f"{source_plugin}_url": url, "current_pipeline": pipeline}
    )

    triggered = []
    monkeypatch.setattr(
        f"router.tasks_pipeline_steps.{next_task}.apply_async",
        lambda args=None, **kwargs: triggered.append((list(args), kwargs)),
    )

    run_task(handle_webhook_task, db_id=db_task.id, source_url=url, via="webhook")

    assert triggered == [([db_task.id, url], {"countdown": CELERY_COUNTDOWN})]


def test_webhook_rejects_an_unknown_pipeline_state(server):
    """Verifies that a bad internal state is caught even if the URL is valid."""
    url = server.task_url(WU_PALMER_PLUGIN)
    db_task = make_router_task(
        data={f"{WU_PALMER_PLUGIN}_url": url, "current_pipeline": "some_unknown_state"}
    )

    result = run_task(
        handle_webhook_task, db_id=db_task.id, source_url=url, via="webhook"
    )

    assert result == "Unrecognized pipeline state"


# --- FULL RUNS ---


def test_full_run_through_both_pipelines(server, inline):
    """Drives the complete state machine with the events the sub-plugins send."""
    db_task = make_router_task(
        selections={"attr1": WU_PALMER_PLUGIN, "attr2": MAPPING_PLUGIN}
    )

    run_inline(start_routing_task, db_task.id)
    for plugin in (WU_PALMER_PLUGIN, TRANSFORMERS_PLUGIN, AGGREGATOR_PLUGIN, MDS_PLUGIN):
        complete_sub_task(db_task, plugin)
    for plugin in (MAPPING_PLUGIN, AGGREGATOR_PLUGIN, MDS_PLUGIN):
        complete_sub_task(db_task, plugin)

    assert inline.errors == []
    assert inline.results == ["All Pipelines Completed Successfully!"]
    assert stored_file_names(db_task) == {
        f"{WU_PALMER_PLUGIN}_mds_vectors.zip",
        f"{MAPPING_PLUGIN}_mds_vectors.zip",
    }
    db_task = reload(db_task)
    assert db_task.progress_value == db_task.progress_target == 8
    assert db_task.data["pipeline_queue"] == []


def test_full_run_with_concatenation_and_pca(server, inline):
    db_task = make_router_task(
        selections={"attr1": WU_PALMER_PLUGIN},
        concatOutput=True,
        reduceDimensions=True,
        pcaDimensions=1,
    )

    run_inline(start_routing_task, db_task.id)
    for plugin in (WU_PALMER_PLUGIN, TRANSFORMERS_PLUGIN, AGGREGATOR_PLUGIN, MDS_PLUGIN):
        complete_sub_task(db_task, plugin)
    complete_sub_task(db_task, VECTOR_CONCAT_PLUGIN)
    complete_sub_task(db_task, PCA_PLUGIN)

    assert inline.errors == []
    assert inline.results == [
        "All Pipelines Completed Successfully And Dimensions Reduced With PCA!"
    ]
    assert stored_file_names(db_task) == {
        "final_vector_pca_reduced.csv",
        "pca_metadata.json",
        "pca_plot.html",
    }


def test_full_run_stops_after_the_concatenation_without_pca(server, inline):
    db_task = make_router_task(selections={"attr1": WU_PALMER_PLUGIN}, concatOutput=True)

    run_inline(start_routing_task, db_task.id)
    for plugin in (WU_PALMER_PLUGIN, TRANSFORMERS_PLUGIN, AGGREGATOR_PLUGIN, MDS_PLUGIN):
        complete_sub_task(db_task, plugin)
    complete_sub_task(db_task, VECTOR_CONCAT_PLUGIN)

    assert inline.errors == []
    assert inline.results == [
        "All Pipelines Completed Successfully And Concatenated Vector Created!"
    ]
    assert stored_file_names(db_task) == {"final_concatenated_vector.csv"}


def test_on_retry_logs_without_app_context():
    """Celery calls ``on_retry`` outside of the task body, i.e. without an app context."""
    db_task = make_router_task()
    # read before the thread starts, the expired attribute would need a session to reload
    db_id = db_task.id

    errors = []

    def call_on_retry():
        # a fresh thread has no Flask app context, just like celery's error handling
        try:
            start_wu_palmer.on_retry(
                Timeout("read timed out"), "sub-task-id", (db_id,), {}, None
            )
        except BaseException as exc:  # noqa: BLE001 - the assertion below reports it
            errors.append(exc)

    thread = threading.Thread(target=call_on_retry)
    thread.start()
    thread.join()

    assert not errors, f"on_retry raised {errors[0]!r}"

    DB.session.expire_all()
    reloaded = ProcessingTask.get_by_id(db_id)
    assert "Transient error" in reloaded.task_log


def test_handle_webhook_synchronization_guard(server, monkeypatch):
    """
    Simulates a race condition by firing 10 simultaneous webhook handlers
    for the exact same source URL to prove the atomic database lock holds up.
    """
    url = server.task_url(WU_PALMER_PLUGIN)
    db_task = make_router_task(
        data={"current_pipeline": WU_PALMER_PLUGIN, f"{WU_PALMER_PLUGIN}_url": url}
    )
    db_id = db_task.id

    triggered = []
    monkeypatch.setattr(
        router_tasks,
        "start_transformers",
        SimpleNamespace(apply_async=lambda *args, **kwargs: triggered.append(kwargs)),
    )

    app = current_app._get_current_object()

    def fire_concurrent_webhook(worker_index):
        with app.app_context():
            return run_task(
                handle_webhook_task,
                db_id=db_id,
                source_url=url,
                via=f"synch_test_{worker_index}",
            )

    # Launch 10 workers simultaneously
    worker_count = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        # executor.map fires all threads at once and collects their return values
        results = list(executor.map(fire_concurrent_webhook, range(worker_count)))

    # Out of 10 concurrent workers, exactly 1 should win, and 9 should be rejected.
    assert (
        len(triggered) == 1
    ), f"Expected exactly 1 progression, but got {len(triggered)}!"

    rejected_count = results.count("Sub-task already progressed")
    assert (
        rejected_count == worker_count - 1
    ), f"Expected 9 rejected workers, but got {rejected_count}. Results: {results}"
