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

import zipfile
from io import BytesIO
from types import SimpleNamespace

import pytest
from requests.exceptions import ConnectionError, HTTPError, Timeout

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from router.schemas import MAPPING_PLUGIN, MDS_PLUGIN, WU_PALMER_PLUGIN
from router.tasks_helpers import (
    CELERY_COUNTDOWN,
    PipelineTask,
    calculate_recommendations,
    extract_output_url,
    has_enough_pca_dimensions,
    is_store_mds_output,
    load_entity_attributes,
    load_task,
    log_task_event,
    plugin_process_url,
    run_pipeline_step,
    save_intermediate_results,
    taxonomy_ref,
)
from router.tasks_pipeline_steps import start_wu_palmer
from router.tests.data import (
    ENTITIES_URL,
    PLUGIN_URLS,
    TAXONOMY_MEMBERS,
    PluginServer,
    input_file_responses,
    make_router_task,
    mock_open_url,
    router_params,
)

from tests.utils import MockResponse

# The helpers touch the database, the file store and ``current_app``, so they
# need the app context that comes with the worker fixture.
pytestmark = pytest.mark.usefixtures("celery_worker")


def _taxonomies_zip() -> zipfile.ZipFile:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in TAXONOMY_MEMBERS.items():
            archive.writestr(name, text)
    return zipfile.ZipFile(buffer)


def _log_of(db_task: ProcessingTask) -> str:
    DB.session.expire_all()
    return ProcessingTask.get_by_id(db_task.id).task_log


# --- OUTPUT / URL RESOLUTION ---


def test_extract_output_url_returns_matching_href():
    outputs = [
        {"dataType": "entity/list", "href": "http://localhost/a"},
        {"dataType": "entity/vector", "href": "http://localhost/b"},
    ]
    assert extract_output_url(outputs, "entity/vector") == "http://localhost/b"


def test_extract_output_url_raises_for_unknown_data_type():
    with pytest.raises(ValueError, match="entity/vector"):
        extract_output_url([{"dataType": "entity/list", "href": "x"}], "entity/vector")


def test_plugin_process_url_resolves_from_stored_metadata_url(monkeypatch):
    monkeypatch.setattr(
        "router.tasks_helpers.get_plugin_endpoint", lambda url: url + "process/"
    )
    db_task = make_router_task()

    assert plugin_process_url(db_task, MDS_PLUGIN) == PLUGIN_URLS[MDS_PLUGIN] + "process/"


# --- TASK LOADING ---


def test_load_task_returns_persisted_task():
    db_task = make_router_task()
    assert load_task(db_task.id).id == db_task.id


def test_load_task_raises_for_unknown_id():
    with pytest.raises(KeyError):
        load_task(123456789)


# --- TASK LOGGING ---


@pytest.mark.parametrize("level", ["info", "warning"])
def test_log_task_event_writes_to_the_task_log(level):
    db_task = make_router_task()

    log_task_event(db_task, f"something {level}", level=level)

    assert f"something {level}" in _log_of(db_task)


# --- METADATA / ENTITY PARSING ---


@pytest.mark.parametrize(
    "ref_target,expected",
    [
        ("taxonomies.zip:t_genre.json", "t_genre.json"),
        ("some/other.zip:nested/t_a.json", "nested/t_a.json"),
        ("people.csv", None),
        ("", None),
        (None, None),
    ],
)
def test_taxonomy_ref_only_accepts_zip_references(ref_target, expected):
    metadata = AttributeMetadata(
        ID="a", attribute_type="a", title="A", ref_target=ref_target
    )
    assert taxonomy_ref(metadata) == expected


def test_load_entity_attributes_reads_the_csv_header(monkeypatch):
    mock_open_url(monkeypatch, input_file_responses())

    assert load_entity_attributes(ENTITIES_URL) == {
        "ID",
        "href",
        "genre",
        "instrumentation",
        "composer",
        "missing_tax",
    }


def test_calculate_recommendations_suggests_mapping_for_mapped_taxonomies():
    assert calculate_recommendations(_taxonomies_zip(), "t_genre.json") == MAPPING_PLUGIN


def test_calculate_recommendations_suggests_wu_palmer_without_mappings():
    assert (
        calculate_recommendations(_taxonomies_zip(), "t_instrumentation.json")
        == WU_PALMER_PLUGIN
    )


def test_calculate_recommendations_falls_back_for_unreadable_taxonomies():
    assert (
        calculate_recommendations(_taxonomies_zip(), "missing.json") == WU_PALMER_PLUGIN
    )


# --- OUTPUT STORAGE DECISIONS ---


@pytest.mark.parametrize(
    "concat_output,include_intermediate,expected",
    [
        (False, False, True),
        (False, True, True),
        (True, False, False),
        (True, True, True),
    ],
)
def test_is_store_mds_output(concat_output, include_intermediate, expected):
    """Without concatenation the MDS vectors are the only result of the run."""
    params = router_params(
        concatOutput=concat_output,
        includeIntermediateResultsInOutput=include_intermediate,
    )
    assert is_store_mds_output(params) is expected


def test_save_intermediate_results_persists_the_file():
    db_task = make_router_task()

    save_intermediate_results(
        db_task, 0, db_task.id, b"payload", "result.zip", "relation/element-distances"
    )

    (stored,) = [
        f
        for f in TaskFile.get_task_result_files(db_task.id)
        if f.file_name == "result.zip"
    ]
    assert stored.file_type == "relation/element-distances"
    assert stored.mimetype == "application/zip"


def test_save_intermediate_results_skips_duplicates_on_retry():
    db_task = make_router_task()
    save_intermediate_results(db_task, 0, db_task.id, b"a", "retry.zip", "custom/data")

    save_intermediate_results(db_task, 1, db_task.id, b"a", "retry.zip", "custom/data")

    assert len(TaskFile.get_task_result_files(db_task.id)) == 1
    assert "Skipping save" in _log_of(db_task)


def test_save_intermediate_results_warns_about_parallel_execution():
    """A duplicate on the first attempt can only come from a second worker."""
    db_task = make_router_task()
    save_intermediate_results(db_task, 0, db_task.id, b"a", "race.zip", "custom/data")

    save_intermediate_results(db_task, 0, db_task.id, b"a", "race.zip", "custom/data")

    assert len(TaskFile.get_task_result_files(db_task.id)) == 1
    assert "RACE CONDITION" in _log_of(db_task)


# --- RUNNING A PIPELINE STEP ---


def test_run_pipeline_step_posts_payload_and_subscribes(monkeypatch):
    server = PluginServer().install(monkeypatch)
    db_task = make_router_task()

    run_pipeline_step(
        db_task.id, db_task, WU_PALMER_PLUGIN, "Wu-Palmer", {"attributes": "attr1"}
    )

    assert server.payload(WU_PALMER_PLUGIN) == {"attributes": "attr1"}
    assert db_task.data[f"{WU_PALMER_PLUGIN}_url"] == server.task_url(WU_PALMER_PLUGIN)

    (subscription,) = server.subscriptions
    assert subscription["result_url"] == server.task_url(WU_PALMER_PLUGIN)
    assert subscription["events"] == ["status"]
    assert subscription["monitor_countdown"] == CELERY_COUNTDOWN


def test_run_pipeline_step_uses_a_loopback_webhook_url(monkeypatch):
    """``localhost`` would resolve to the sub-plugin's own container."""
    server = PluginServer().install(monkeypatch)
    db_task = make_router_task()

    run_pipeline_step(db_task.id, db_task, WU_PALMER_PLUGIN, "Wu-Palmer", {})

    (subscription,) = server.subscriptions
    assert subscription["webhook_url"].startswith("http://127.0.0.1")
    assert (
        subscription["monitor_webhook_url"]
        == subscription["webhook_url"] + "?via=watchdog"
    )


def test_run_pipeline_step_appends_watchdog_flag_to_existing_query(monkeypatch):
    server = PluginServer().install(monkeypatch)
    db_task = make_router_task(data={"webhook_url": "http://localhost/hook?db_id=1"})

    run_pipeline_step(db_task.id, db_task, WU_PALMER_PLUGIN, "Wu-Palmer", {})

    (subscription,) = server.subscriptions
    assert (
        subscription["monitor_webhook_url"]
        == "http://127.0.0.1/hook?db_id=1&via=watchdog"
    )


def test_run_pipeline_step_does_not_start_a_second_sub_task(monkeypatch):
    """A retry of the step must not spawn a duplicate sub-task."""
    server = PluginServer().install(monkeypatch)
    existing_url = "http://localhost:5005/plugins/wu-palmer/tasks/999/"
    db_task = make_router_task(data={f"{WU_PALMER_PLUGIN}_url": existing_url})

    run_pipeline_step(db_task.id, db_task, WU_PALMER_PLUGIN, "Wu-Palmer", {})

    assert server.post_count(WU_PALMER_PLUGIN) == 0
    assert server.subscriptions[0]["result_url"] == existing_url


def test_run_pipeline_step_continues_when_subscription_is_rejected(monkeypatch):
    """The polling watchdog still finishes the pipeline without a webhook."""
    PluginServer(subscribe_result=False).install(monkeypatch)
    db_task = make_router_task()

    run_pipeline_step(db_task.id, db_task, WU_PALMER_PLUGIN, "Wu-Palmer", {})

    assert "relying on the polling watchdog" in _log_of(db_task)


def test_run_pipeline_step_continues_when_subscription_raises(monkeypatch):
    PluginServer().install(monkeypatch)
    db_task = make_router_task()

    def _boom(**kwargs):
        raise ConnectionError("no route to host")

    monkeypatch.setattr("router.tasks_helpers.subscribe", _boom)

    run_pipeline_step(db_task.id, db_task, WU_PALMER_PLUGIN, "Wu-Palmer", {})

    assert "relying on the polling watchdog" in _log_of(db_task)
    assert db_task.data[f"{WU_PALMER_PLUGIN}_url"]


def test_run_pipeline_step_raises_when_the_plugin_rejects_the_payload(monkeypatch):
    PluginServer().install(monkeypatch)
    db_task = make_router_task()
    monkeypatch.setattr(
        "requests.post",
        lambda url, **kwargs: MockResponse(url, "text/html", status_code=500),
    )

    with pytest.raises(HTTPError):
        run_pipeline_step(db_task.id, db_task, WU_PALMER_PLUGIN, "Wu-Palmer", {})

    assert f"{WU_PALMER_PLUGIN}_url" not in db_task.data


# --- PIPELINE TASK ERROR HANDLING ---


def test_only_transient_errors_are_retried():
    assert set(PipelineTask.autoretry_for) == {ConnectionError, Timeout}
    assert PipelineTask.retry_backoff is True
    assert PipelineTask.max_retries == 5


@pytest.mark.parametrize(
    "args,kwargs,expected",
    [((), {"db_id": 7}, 7), ((9,), {}, 9), ((), {}, None)],
)
def test_get_db_id_reads_the_task_arguments(args, kwargs, expected):
    assert PipelineTask._get_db_id(args, kwargs) == expected


def test_on_failure_reports_the_error_to_the_processing_task(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "router.tasks_helpers.save_task_error",
        SimpleNamespace(delay=lambda **kwargs: recorded.append(kwargs)),
    )

    start_wu_palmer.on_failure(RuntimeError("boom"), "celery-id", (42,), {}, None)

    assert recorded == [{"failing_task_id": "celery-id", "db_id": 42}]


# Not sure that this is the right behavior, but it is what the code currently does.
def test_on_failure_without_db_id_does_not_report_an_unknown_task(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "router.tasks_helpers.save_task_error",
        SimpleNamespace(delay=lambda **kwargs: recorded.append(kwargs)),
    )

    start_wu_palmer.on_failure(RuntimeError("boom"), "celery-id", (), {}, None)

    assert recorded == []


def test_on_retry_is_recorded_in_the_task_log():
    db_task = make_router_task()

    start_wu_palmer.on_retry(Timeout("timed out"), "celery-id", (db_task.id,), {}, None)

    assert "Retry execution" in _log_of(db_task)


# --- PCA DIMENSION CHECK ---


def _pca_params(**overrides):
    return router_params(concatOutput=True, reduceDimensions=True, **overrides)


def _silent_task():
    return SimpleNamespace(add_task_log_entry=lambda *args, **kwargs: None)


def test_has_enough_pca_dimensions_returns_true_when_vector_has_more_dimensions_than_requested():
    vector_response = MockResponse(
        "http://localhost/vector.csv", "text/csv", text="ID,dim0,dim1,dim2\nentA,1,2,3\n"
    )

    assert (
        has_enough_pca_dimensions(
            _silent_task(), _pca_params(pcaDimensions=2), vector_response
        )
        is True
    )


def test_has_enough_pca_dimensions_returns_false_when_requested_dimensions_match_vector():
    vector_response = MockResponse(
        "http://localhost/vector.csv", "text/csv", text="ID,dim0,dim1,dim2\nentA,1,2,3\n"
    )

    assert (
        has_enough_pca_dimensions(
            _silent_task(), _pca_params(pcaDimensions=3), vector_response
        )
        is False
    )


def test_has_enough_pca_dimensions_returns_false_for_an_empty_vector_file():
    vector_response = MockResponse(
        "http://localhost/vector.csv", "text/csv", text="ID,dim0\n"
    )

    assert (
        has_enough_pca_dimensions(
            _silent_task(), _pca_params(pcaDimensions=1), vector_response
        )
        is False
    )


def test_has_enough_pca_dimensions_returns_false_without_a_content_type():
    vector_response = MockResponse("http://localhost/vector", "", text=".")

    assert (
        has_enough_pca_dimensions(
            _silent_task(), _pca_params(pcaDimensions=1), vector_response
        )
        is False
    )
