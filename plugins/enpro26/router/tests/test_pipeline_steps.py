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

"""Tests for the individual pipeline steps.

The execution of a step itself is covered by the ``run_pipeline_step`` tests in
``test_tasks_helpers.py``. What is verified here is the extra logic of each
step: the payload it builds (checked against the schema of the plugin that
receives it), which intermediate results it stores, and how it advances the
pipeline state.
"""

from importlib import import_module
from types import SimpleNamespace

import pytest
from marshmallow import EXCLUDE

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile
from qhana_plugin_runner.storage import STORE
from router.schemas import (
    AGGREGATOR_PLUGIN,
    FINALIZE_PIPELINE,
    MAPPING_PLUGIN,
    MDS_PLUGIN,
    PCA_PLUGIN,
    TRANSFORMERS_PLUGIN,
    VECTOR_CONCAT_PLUGIN,
    WU_PALMER_PLUGIN,
)
from router.tasks_pipeline_steps import (
    OUTPUT_FORMATS,
    PCA_DEFAULTS,
    finalize_pca,
    finalize_pipeline,
    finalize_vector_concat,
    launch_next_pipeline,
    start_aggregator,
    start_mapping,
    start_mds,
    start_pca,
    start_transformers,
    start_vector_concat,
    start_wu_palmer,
)
from router.tests.data import (
    ENTITIES_URL,
    METADATA_URL,
    TAXONOMIES_URL,
    PluginServer,
    capture_pipeline_step,
    make_router_task,
)

from tests.utils import run_task

pytestmark = pytest.mark.usefixtures("celery_worker")

# The schema each sub-plugin uses to load the posted form data.
PLUGIN_SCHEMAS = {
    WU_PALMER_PLUGIN: ("wu_palmer", "InputParametersSchema"),
    MAPPING_PLUGIN: ("mapping_distances.schemas", "InputParametersSchema"),
    TRANSFORMERS_PLUGIN: ("transformer.schemas", "InputParametersSchema"),
    AGGREGATOR_PLUGIN: ("aggregator.schemas", "InputParametersSchema"),
    MDS_PLUGIN: ("attribute_mds.schemas", "InputParametersSchema"),
    VECTOR_CONCAT_PLUGIN: ("vector_concat.schemas", "VectorConcatSchema"),
    PCA_PLUGIN: ("pca.schemas", "InputParametersSchema"),
}


@pytest.fixture
def server(monkeypatch) -> PluginServer:
    return PluginServer().install(monkeypatch)


@pytest.fixture
def steps(monkeypatch) -> list:
    return capture_pipeline_step(monkeypatch)


@pytest.fixture
def dispatched(monkeypatch) -> list:
    """Record follow-up dispatches instead of running them on the worker."""
    calls = []

    def _recorder(name):
        def _apply_async(args=None, **kwargs):
            calls.append((name, list(args or [])))

        return _apply_async

    for name in ("start_wu_palmer", "start_mapping", "start_vector_concat", "start_pca"):
        monkeypatch.setattr(
            f"router.tasks_pipeline_steps.{name}.apply_async", _recorder(name)
        )

    monkeypatch.setattr(
        "router.tasks_pipeline_steps.save_task_result",
        SimpleNamespace(
            delay=lambda *args: calls.append(("save_task_result", list(args)))
        ),
    )
    return calls


def payload_for(steps: list, plugin: str) -> dict:
    for call in steps:
        if call["plugin_name"] == plugin:
            return call["payload"]
    raise AssertionError(f"No pipeline step started for {plugin}: {steps}")


def assert_payload_matches_plugin_schema(plugin: str, payload: dict):
    """The receiving plugin loads the form with ``unknown=EXCLUDE``.

    A key that does not match a ``data_key`` is therefore dropped without any
    error, which is why the unknown keys are rejected explicitly here.
    """
    module_name, schema_name = PLUGIN_SCHEMAS[plugin]
    schema_class = getattr(import_module(module_name), schema_name)
    schema = schema_class()

    known_keys = {field.data_key for field in schema.fields.values()}
    assert (
        set(payload) <= known_keys
    ), f"{plugin} would silently ignore {set(payload) - known_keys}"

    schema_class(unknown=EXCLUDE).load(payload)


def stored_files(db_task: ProcessingTask) -> dict:
    DB.session.expire_all()
    return {f.file_name: f for f in TaskFile.get_task_result_files(db_task.id)}


def reload(db_task: ProcessingTask) -> ProcessingTask:
    DB.session.expire_all()
    return ProcessingTask.get_by_id(db_task.id)


# --- PAYLOADS ---


def test_wu_palmer_payload(server, steps):
    db_task = make_router_task()

    run_task(start_wu_palmer, db_id=db_task.id)

    payload = payload_for(steps, WU_PALMER_PLUGIN)
    assert payload == {
        "entitiesUrl": ENTITIES_URL,
        "entitiesMetadataUrl": METADATA_URL,
        "taxonomiesZipUrl": TAXONOMIES_URL,
        "attributes": "attr1",
        "rootIsPartOfHierarchy": "false",
    }
    assert_payload_matches_plugin_schema(WU_PALMER_PLUGIN, payload)


def test_wu_palmer_payload_forwards_the_hierarchy_setting(server, steps):
    db_task = make_router_task(rootIsPartOfHierarchy=True)

    run_task(start_wu_palmer, db_id=db_task.id)

    assert payload_for(steps, WU_PALMER_PLUGIN)["rootIsPartOfHierarchy"] == "true"


def test_mapping_payload(server, steps):
    db_task = make_router_task(distanceMetric="cosine")

    run_task(start_mapping, db_id=db_task.id)

    payload = payload_for(steps, MAPPING_PLUGIN)
    assert payload == {
        "entitiesUrl": ENTITIES_URL,
        "entitiesMetadataUrl": METADATA_URL,
        "taxonomiesZipUrl": TAXONOMIES_URL,
        "attributes": "attr2",
        "distanceMetric": "cosine",
    }
    assert_payload_matches_plugin_schema(MAPPING_PLUGIN, payload)


def test_transformers_payload(server, steps):
    db_task = make_router_task(transformer="gaussian_inverse")

    run_task(
        start_transformers,
        db_id=db_task.id,
        source_url=server.task_url(WU_PALMER_PLUGIN),
    )

    payload = payload_for(steps, TRANSFORMERS_PLUGIN)
    assert payload == {
        "similaritiesUrl": server.output_url(
            WU_PALMER_PLUGIN, "relation/element-similarities"
        ),
        "attributes": "attr1",
        "transformer": "gaussian_inverse",
    }
    assert_payload_matches_plugin_schema(TRANSFORMERS_PLUGIN, payload)


def test_aggregator_payload(server, steps):
    db_task = make_router_task()

    run_task(
        start_aggregator,
        db_id=db_task.id,
        source_url=server.task_url(TRANSFORMERS_PLUGIN),
    )

    payload = payload_for(steps, AGGREGATOR_PLUGIN)
    assert payload == {
        "entitiesUrl": ENTITIES_URL,
        "elementDistancesUrl": server.output_url(
            TRANSFORMERS_PLUGIN, "relation/element-distances"
        ),
    }
    assert_payload_matches_plugin_schema(AGGREGATOR_PLUGIN, payload)


def test_mds_payload(server, steps):
    db_task = make_router_task(
        mdsDimensions=3, metric="nonmetric_mds", nInit=8, maxIter=42
    )

    run_task(start_mds, db_id=db_task.id, source_url=server.task_url(AGGREGATOR_PLUGIN))

    payload = payload_for(steps, MDS_PLUGIN)
    assert payload == {
        "attributeDistancesUrl": server.output_url(
            AGGREGATOR_PLUGIN, "relation/attribute-distances"
        ),
        "dimensions": 3,
        "metric": "nonmetric_mds",
        "nInit": 8,
        "maxIter": 42,
        "missingDataHandling": "mean",
    }
    assert_payload_matches_plugin_schema(MDS_PLUGIN, payload)


def test_vector_concat_payload(server, steps):
    urls = [
        "http://localhost:5005/files/10/a.zip",
        "http://localhost:5005/files/11/b.zip",
    ]
    db_task = make_router_task(
        concatOutput=True, outputFormat="json", data={"vector_zip_urls": urls}
    )

    run_task(start_vector_concat, db_id=db_task.id)

    payload = payload_for(steps, VECTOR_CONCAT_PLUGIN)
    assert payload == {
        "urls": "\n".join(urls),
        "outputFormat": "json",
        "outputSuffix": "final_concatenated_vector",
    }
    assert_payload_matches_plugin_schema(VECTOR_CONCAT_PLUGIN, payload)


def test_pca_payload(server, steps):
    vector_url = server.output_url(VECTOR_CONCAT_PLUGIN, "entity/vector")
    db_task = make_router_task(
        concatOutput=True,
        reduceDimensions=True,
        pcaType="kernel",
        pcaDimensions=2,
        solver="randomized",
        tol=0.5,
        iteratedPower=7,
    )

    run_task(start_pca, db_id=db_task.id, vector_url=vector_url)

    payload = payload_for(steps, PCA_PLUGIN)
    assert payload == {
        **PCA_DEFAULTS,
        "entityPointsUrl": vector_url,
        "pcaType": "kernel",
        "dimensions": 2,
        "solver": "randomized",
        "tol": 0.5,
        "iteratedPower": 7,
    }
    assert_payload_matches_plugin_schema(PCA_PLUGIN, payload)


def test_pca_defaults_cover_every_parameter_the_form_omits(app):
    """The PCA schema has no defaults, so the router has to send every field."""
    schema = getattr(import_module("pca.schemas"), "InputParametersSchema")()
    required = {field.data_key for field in schema.fields.values() if field.required}
    sent_by_router = set(PCA_DEFAULTS) | {
        "entityPointsUrl",
        "pcaType",
        "dimensions",
        "solver",
        "tol",
        "iteratedPower",
    }
    assert required <= sent_by_router


# --- INTERMEDIATE RESULTS ---


def test_steps_store_no_intermediate_results_by_default(server, steps):
    db_task = make_router_task()

    run_task(
        start_transformers,
        db_id=db_task.id,
        source_url=server.task_url(WU_PALMER_PLUGIN),
    )

    assert stored_files(db_task) == {}


# --- CALLING THE SUB-PLUGINS ---

# (pipeline, step, plugin that ran before, data type consumed, payload key,
#  plugin the step starts, name of the stored intermediate result)
PIPELINE_CHAIN = [
    (
        WU_PALMER_PLUGIN,
        start_transformers,
        WU_PALMER_PLUGIN,
        "relation/element-similarities",
        "similaritiesUrl",
        TRANSFORMERS_PLUGIN,
        "wu_palmer_similarities.zip",
    ),
    (
        WU_PALMER_PLUGIN,
        start_aggregator,
        TRANSFORMERS_PLUGIN,
        "relation/element-distances",
        "elementDistancesUrl",
        AGGREGATOR_PLUGIN,
        "wu_palmer_element_distances.zip",
    ),
    (
        WU_PALMER_PLUGIN,
        start_mds,
        AGGREGATOR_PLUGIN,
        "relation/attribute-distances",
        "attributeDistancesUrl",
        MDS_PLUGIN,
        "wu_palmer_attribute_distances.zip",
    ),
    (
        MAPPING_PLUGIN,
        start_aggregator,
        MAPPING_PLUGIN,
        "relation/element-distances",
        "elementDistancesUrl",
        AGGREGATOR_PLUGIN,
        "mapping_element_distances.zip",
    ),
    (
        MAPPING_PLUGIN,
        start_mds,
        AGGREGATOR_PLUGIN,
        "relation/attribute-distances",
        "attributeDistancesUrl",
        MDS_PLUGIN,
        "mapping_attribute_distances.zip",
    ),
]


@pytest.mark.parametrize(
    "pipeline,step,source_plugin,data_type,payload_key,started_plugin,file_name",
    PIPELINE_CHAIN,
    ids=[
        "wu_palmer: wu-palmer -> transformers",
        "wu_palmer: transformers -> aggregator",
        "wu_palmer: aggregator -> mds",
        "mapping: mapping -> aggregator",
        "mapping: aggregator -> mds",
    ],
)
def test_step_hands_the_previous_output_to_the_next_plugin(
    server,
    pipeline,
    step,
    source_plugin,
    data_type,
    payload_key,
    started_plugin,
    file_name,
):
    """Every step of both pipelines, from the predecessor's result to the next call."""
    db_task = make_router_task(
        includeIntermediateResultsInOutput=True, data={"current_pipeline": pipeline}
    )

    run_task(step, db_id=db_task.id, source_url=server.task_url(source_plugin))

    # the step picks the output it needs out of the predecessor's results ...
    assert server.post_count(started_plugin) == 1
    assert server.payload(started_plugin)[payload_key] == server.output_url(
        source_plugin, data_type
    )
    # ... remembers where the started sub-task will report back ...
    started_url = server.task_url(started_plugin)
    assert reload(db_task).data[f"{started_plugin}_url"] == started_url
    assert server.subscriptions[-1]["result_url"] == started_url
    # ... and keeps the consumed data as an intermediate result
    stored = stored_files(db_task)[file_name]
    assert stored.file_type == data_type
    assert stored.mimetype == "application/zip"


def test_missing_plugin_output_fails_the_step(server, steps):
    """The mapping task publishes distances, not the expected similarities."""
    db_task = make_router_task()

    with pytest.raises(ValueError, match="relation/element-similarities"):
        run_task(
            start_transformers,
            db_id=db_task.id,
            source_url=server.task_url(MAPPING_PLUGIN),
        )


# --- FINALIZING A SINGLE PIPELINE ---


def test_finalize_pipeline_stores_the_mds_vectors(server, dispatched):
    db_task = make_router_task(data={"current_pipeline": WU_PALMER_PLUGIN})

    run_task(finalize_pipeline, db_id=db_task.id, source_url=server.task_url(MDS_PLUGIN))

    stored = stored_files(db_task)[f"{WU_PALMER_PLUGIN}_mds_vectors.zip"]
    assert stored.file_type == "entity/vector"


def test_finalize_pipeline_skips_the_mds_vectors_when_concatenating(server, dispatched):
    """With concatenation the MDS vectors are only an intermediate result."""
    db_task = make_router_task(
        concatOutput=True, data={"current_pipeline": WU_PALMER_PLUGIN}
    )

    run_task(finalize_pipeline, db_id=db_task.id, source_url=server.task_url(MDS_PLUGIN))

    assert stored_files(db_task) == {}


def test_finalize_pipeline_collects_the_vector_urls_for_the_concatenation(
    server, dispatched
):
    db_task = make_router_task(
        concatOutput=True, data={"current_pipeline": WU_PALMER_PLUGIN}
    )

    run_task(finalize_pipeline, db_id=db_task.id, source_url=server.task_url(MDS_PLUGIN))

    assert reload(db_task).data["vector_zip_urls"] == [
        server.output_url(MDS_PLUGIN, "entity/vector")
    ]


def test_finalize_pipeline_starts_the_next_pipeline(server, dispatched):
    db_task = make_router_task(data={"pipeline_queue": [MAPPING_PLUGIN]})

    run_task(finalize_pipeline, db_id=db_task.id, source_url=server.task_url(MDS_PLUGIN))

    assert ("start_mapping", [db_task.id]) in dispatched
    assert reload(db_task).data["current_pipeline"] == MAPPING_PLUGIN


# --- PIPELINE ORCHESTRATION ---


def test_launch_next_pipeline_pops_the_queue(dispatched):
    db_task = make_router_task(
        data={"pipeline_queue": [WU_PALMER_PLUGIN, MAPPING_PLUGIN]}
    )

    launch_next_pipeline(db_task)

    assert db_task.data["current_pipeline"] == WU_PALMER_PLUGIN
    assert db_task.data["pipeline_queue"] == [MAPPING_PLUGIN]
    assert ("start_wu_palmer", [db_task.id]) in dispatched


def test_launch_next_pipeline_resets_the_reused_step_urls(dispatched):
    """The transformer, aggregator and MDS steps run once per pipeline."""
    stale = {
        f"{plugin}_url": "http://localhost/old/"
        for plugin in (
            WU_PALMER_PLUGIN,
            MAPPING_PLUGIN,
            TRANSFORMERS_PLUGIN,
            AGGREGATOR_PLUGIN,
            MDS_PLUGIN,
        )
    }
    db_task = make_router_task(
        data={
            "pipeline_queue": [MAPPING_PLUGIN],
            "progressed_via": {"a": "webhook"},
            "webhook_seen": {"a": "2026-01-01"},
            **stale,
        }
    )

    launch_next_pipeline(db_task)

    assert not [key for key in stale if key in db_task.data]
    assert db_task.data["progressed_via"] == {}
    assert db_task.data["webhook_seen"] == {}


def test_launch_next_pipeline_rejects_a_pipeline_without_a_start_step(dispatched):
    """Without a start step no webhook can arrive, so the task would stall silently."""
    db_task = make_router_task(data={"pipeline_queue": ["one_hot"]})

    with pytest.raises(ValueError, match="No pipeline start step for 'one_hot'"):
        launch_next_pipeline(db_task)

    assert dispatched == []


def test_launch_next_pipeline_starts_the_concatenation_when_the_queue_is_empty(
    dispatched,
):
    db_task = make_router_task(concatOutput=True, data={"pipeline_queue": []})

    launch_next_pipeline(db_task)

    assert db_task.data["current_pipeline"] == FINALIZE_PIPELINE
    assert ("start_vector_concat", [db_task.id]) in dispatched


def test_launch_next_pipeline_finishes_the_task_without_concatenation(dispatched):
    db_task = make_router_task(data={"pipeline_queue": []})

    launch_next_pipeline(db_task)

    assert (
        "save_task_result",
        ["All Pipelines Completed Successfully!", db_task.id],
    ) in dispatched


def test_launch_next_pipeline_warns_about_duplicate_outputs(dispatched):
    db_task = make_router_task(data={"pipeline_queue": []})
    for _ in range(2):
        STORE.persist_task_result(
            db_task.id, b"x", "duplicate.zip", "entity/vector", "application/zip"
        )

    launch_next_pipeline(db_task)

    assert "Output contains duplicates: ['duplicate.zip']" in reload(db_task).task_log


# --- VECTOR CONCATENATION AND PCA ---


@pytest.mark.parametrize("output_format", sorted(OUTPUT_FORMATS))
def test_finalize_vector_concat_stores_the_final_vector(
    server, dispatched, output_format
):
    extension, mimetype = OUTPUT_FORMATS[output_format]
    db_task = make_router_task(concatOutput=True, outputFormat=output_format)

    run_task(
        finalize_vector_concat,
        db_id=db_task.id,
        source_url=server.task_url(VECTOR_CONCAT_PLUGIN),
    )

    stored = stored_files(db_task)[f"final_concatenated_vector{extension}"]
    assert stored.file_type == "entity/vector"
    assert stored.mimetype == mimetype
    assert (
        "save_task_result",
        [
            "All Pipelines Completed Successfully And Concatenated Vector Created!",
            db_task.id,
        ],
    ) in dispatched


def test_finalize_vector_concat_starts_pca_when_requested(server, dispatched):
    db_task = make_router_task(concatOutput=True, reduceDimensions=True, pcaDimensions=1)

    run_task(
        finalize_vector_concat,
        db_id=db_task.id,
        source_url=server.task_url(VECTOR_CONCAT_PLUGIN),
    )

    vector_url = server.output_url(VECTOR_CONCAT_PLUGIN, "entity/vector")
    assert ("start_pca", [db_task.id, vector_url]) in dispatched
    assert "final_concatenated_vector.csv" not in stored_files(db_task)


def test_finalize_vector_concat_keeps_the_unreduced_vector_on_request(server, dispatched):
    db_task = make_router_task(
        concatOutput=True,
        reduceDimensions=True,
        pcaDimensions=1,
        includeIntermediateResultsInOutput=True,
    )

    run_task(
        finalize_vector_concat,
        db_id=db_task.id,
        source_url=server.task_url(VECTOR_CONCAT_PLUGIN),
    )

    assert "concatenated_vector.csv" in stored_files(db_task)


def test_finalize_vector_concat_skips_pca_without_enough_dimensions(server, dispatched):
    """The stub vector has three dimensions, so reducing to three is pointless."""
    db_task = make_router_task(concatOutput=True, reduceDimensions=True, pcaDimensions=3)

    run_task(
        finalize_vector_concat,
        db_id=db_task.id,
        source_url=server.task_url(VECTOR_CONCAT_PLUGIN),
    )

    assert not [call for call in dispatched if call[0] == "start_pca"]
    assert "final_concatenated_vector.csv" in stored_files(db_task)


def test_finalize_pca_stores_all_results(server, dispatched):
    db_task = make_router_task(concatOutput=True, reduceDimensions=True)

    run_task(finalize_pca, db_id=db_task.id, source_url=server.task_url(PCA_PLUGIN))

    stored = stored_files(db_task)
    assert stored["final_vector_pca_reduced.csv"].file_type == "entity/vector"
    assert stored["pca_metadata.json"].file_type == "custom/pca-metadata"
    assert stored["pca_plot.html"].file_type == "custom/plot"
    assert (
        "save_task_result",
        [
            "All Pipelines Completed Successfully And Dimensions Reduced With PCA!",
            db_task.id,
        ],
    ) in dispatched


def test_finalize_pca_works_without_a_plot(server, dispatched):
    """The PCA plugin only plots results with at most three dimensions."""
    db_task = make_router_task(concatOutput=True, reduceDimensions=True)
    outputs = server.outputs(PCA_PLUGIN)
    outputs[:] = [o for o in outputs if o["dataType"] != "custom/plot"]

    run_task(finalize_pca, db_id=db_task.id, source_url=server.task_url(PCA_PLUGIN))

    stored = stored_files(db_task)
    assert "pca_plot.html" not in stored
    assert "final_vector_pca_reduced.csv" in stored
