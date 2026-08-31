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

import re
from http import HTTPStatus
from json import loads
from urllib.parse import urlsplit

import pytest
from flask import url_for

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from router import ROUTER_BLP, Router
from router.routes import INPUT_FIELD_GROUPS
from router.schemas import (
    MAPPING_PLUGIN,
    NONE_PLUGIN,
    PIPELINE_OPTIONS,
    PIPELINE_PLUGINS,
    WU_PALMER_PLUGIN,
    InputParametersSchema,
)
from router.tests.data import router_payload

from tests.utils import mock_task_dispatch


def _path(endpoint: str, **kwargs) -> str:
    return urlsplit(url_for(f"{ROUTER_BLP.name}.{endpoint}", **kwargs)).path


def _form(**overrides) -> dict:
    """The input parameters as a browser would submit them."""
    payload = router_payload(**overrides)
    return {
        key: ("true" if value else "false") if isinstance(value, bool) else value
        for key, value in payload.items()
    }


def test_metadata_endpoint_returns_full_descriptor(client):
    resp = client.get(url_for(f"{ROUTER_BLP.name}.PluginsView"))
    plugin = Router.instance

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_json()

    assert body["name"] == plugin.name
    assert body["type"] == "processing"
    assert len(body["entryPoint"]["dataOutput"]) == 8


def test_microfrontend_renders_form_fields(client):
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))
    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)

    assert "Entities URL" in body
    assert "Distance Metric" in body
    assert "Transformer" in body
    assert "Metric" in body
    assert "Concat output" in body
    assert "Output Format" in body


def test_process_valid_payload_redirects_to_task(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    valid_payload = {
        "entitiesUrl": "http://example.com/data.csv",
        "entitiesMetadataUrl": "http://example.com/meta.json",
        "taxonomiesZipUrl": "http://example.com/tax.zip",
        "distanceMetric": "euclidean",
        "transformer": "linear_inverse",
        "mdsDimensions": 2,
        "metric": "metric_mds",
        "nInit": 4,
        "maxIter": 300,
        "missingDataHandling": "mean",
        "reduceDimensions": False,
        "pcaType": "normal",
        "pcaDimensions": 1,
        "solver": "auto",
        "tol": 0,
        "iteratedPower": 0,
    }

    resp = client.post(url_for(f"{ROUTER_BLP.name}.ProcessView"), data=valid_payload)

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert re.fullmatch(r"/tasks/\d+/", urlsplit(resp.headers["Location"]).path)


def test_routing_step_frontend_renders_attribute_dropdowns(client):
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["instrumentation", "genre"]
    db_task.data["recommendations"] = {"genre": PIPELINE_OPTIONS[MAPPING_PLUGIN]}
    db_task.save(commit=True)

    resp = client.get(_path("RoutingStepFrontend", db_id=db_task.id))

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)
    assert 'name="pipeline_instrumentation"' in body
    assert 'name="pipeline_genre"' in body
    assert PIPELINE_OPTIONS[WU_PALMER_PLUGIN] in body
    assert PIPELINE_OPTIONS[MAPPING_PLUGIN] in body


def test_routing_step_process_records_selection_and_redirects(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = ProcessingTask(
        task_name="router_test",
        parameters='{"entities_url": "http://example.com/data.csv"}',
    )
    db_task.data["taxonomy_attributes"] = ["instrumentation", "genre"]
    db_task.save(commit=True)

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={
            "pipeline_instrumentation": WU_PALMER_PLUGIN,
            "pipeline_genre": MAPPING_PLUGIN,
        },
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert re.fullmatch(r"/tasks/\d+/", urlsplit(resp.headers["Location"]).path)

    db_task = ProcessingTask.get_by_id(db_task.id)

    assert db_task.data.get("routing_selections") == {
        "instrumentation": WU_PALMER_PLUGIN,
        "genre": MAPPING_PLUGIN,
    }


def test_webhook_view_accepts_status_events(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    # Create a dummy task in DB
    db_task = ProcessingTask(task_name="router_test")
    db_task.save(commit=True)

    resp = client.post(
        url_for(f"{ROUTER_BLP.name}.WebhookView", db_id=db_task.id),
        query_string={"source": "http://localhost/tasks/1/", "event": "status"},
    )

    assert resp.status_code == HTTPStatus.OK
    assert resp.get_data(as_text=True) == '"Webhook received"\n'


def test_metadata_data_inputs_match_the_input_schema(client):
    """A data input the UI cannot map to a form field would break the workflow."""
    resp = client.get(url_for(f"{ROUTER_BLP.name}.PluginsView"))
    data_input = resp.get_json()["entryPoint"]["dataInput"]
    form_keys = {field.data_key for field in InputParametersSchema().fields.values()}

    assert len(data_input) == 3
    assert {entry["parameter"] for entry in data_input} <= form_keys


def test_microfrontend_renders_every_field_group(client):
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))
    body = resp.get_data(as_text=True)

    for title, _field_names, _expanded in INPUT_FIELD_GROUPS:
        assert title in body


def test_microfrontend_prefills_the_defaults(client):
    """The MDS and PCA parameters are prefilled so the form can be submitted as is."""
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))
    body = resp.get_data(as_text=True)

    assert 'value="300"' in body  # maxIter
    assert 'value="4"' in body  # nInit


def test_microfrontend_reports_pca_without_concatenation(client):
    """The cross field rule has to reach the user before the task is started."""
    resp = client.post(
        url_for(f"{ROUTER_BLP.name}.MicroFrontend"),
        data=_form(concatOutput=False, reduceDimensions=True),
    )

    assert resp.status_code == HTTPStatus.OK
    assert "has to be enabled as well" in resp.get_data(as_text=True)


def test_microfrontend_accepts_an_empty_form(client):
    """The frontend is validated partially, an untouched form is not an error."""
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))

    assert resp.status_code == HTTPStatus.OK
    assert "qhana-error-message" not in resp.get_data(as_text=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"entitiesUrl": "not-a-url"},
        {"mdsDimensions": 0},
        {"transformer": "no_such_transformer"},
        {"outputFormat": "xml"},
        {"concatOutput": False, "reduceDimensions": True},
    ],
)
def test_process_rejects_invalid_input(client, monkeypatch, overrides):
    mock_task_dispatch(monkeypatch)

    resp = client.post(url_for(f"{ROUTER_BLP.name}.ProcessView"), data=_form(**overrides))

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_routing_step_frontend_shows_the_inputs_of_the_first_step(client):
    db_task = ProcessingTask(
        task_name="router_test",
        parameters='{"entitiesUrl": "http://example.com/step-one.csv"}',
    )
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.save(commit=True)

    body = client.get(_path("RoutingStepFrontend", db_id=db_task.id)).get_data(
        as_text=True
    )

    assert "http://example.com/step-one.csv" in body


def test_routing_step_frontend_without_attributes(client):
    """Preprocessing may not find any taxonomy attribute at all."""
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.save(commit=True)

    resp = client.get(_path("RoutingStepFrontend", db_id=db_task.id))

    assert resp.status_code == HTTPStatus.OK
    assert 'name="pipeline_' not in resp.get_data(as_text=True)


def test_routing_step_frontend_rejects_an_unknown_task(client):
    with pytest.raises(KeyError):
        client.get(_path("RoutingStepFrontend", db_id=999999))


def test_routing_step_clears_the_previous_step(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.save(commit=True)
    db_task.add_next_step(
        href="http://localhost/step/",
        ui_href="http://localhost/step-ui/",
        step_id="routing-step",
    )
    db_task.save(commit=True)

    client.post(
        _path("RoutingStepView", db_id=db_task.id),
        data={"pipeline_genre": WU_PALMER_PLUGIN},
    )

    assert not ProcessingTask.get_by_id(db_task.id).has_uncleared_step


def test_routing_step_rejects_a_routing_without_any_pipeline(client, monkeypatch):
    """Routing every attribute to ``None`` would leave the pipeline queue empty."""
    mock_task_dispatch(monkeypatch)
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.save(commit=True)

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id), data={"pipeline_genre": NONE_PLUGIN}
    )

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_routing_step_rejects_an_unknown_pipeline(client, monkeypatch):
    mock_task_dispatch(monkeypatch)
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.save(commit=True)

    resp = client.post(
        _path("RoutingStepView", db_id=db_task.id), data={"pipeline_genre": "quantum"}
    )

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_routing_step_rejects_an_unknown_task(client, monkeypatch):
    mock_task_dispatch(monkeypatch)

    with pytest.raises(KeyError):
        client.post(
            _path("RoutingStepView", db_id=999999),
            data={"pipeline_genre": WU_PALMER_PLUGIN},
        )


@pytest.fixture
def webhook_dispatch(monkeypatch):
    dispatched = []
    monkeypatch.setattr(
        "router.routes.handle_webhook_task.apply_async",
        lambda **kwargs: dispatched.append(kwargs),
    )
    return dispatched


def test_webhook_forwards_the_delivery_method(client, webhook_dispatch):
    db_task = ProcessingTask(task_name="router_test")
    db_task.save(commit=True)

    client.post(
        _path("WebhookView", db_id=db_task.id),
        query_string={
            "source": "http://localhost/tasks/1/",
            "event": "status",
            "via": "watchdog",
        },
    )

    assert webhook_dispatch == [
        {
            "kwargs": {
                "db_id": db_task.id,
                "source_url": "http://localhost/tasks/1/",
                "via": "watchdog",
            },
            "countdown": 2,
        }
    ]


@pytest.mark.parametrize(
    "query_string",
    [
        {"event": "status"},
        {"source": "http://localhost/tasks/1/"},
        {"source": "http://localhost/tasks/1/", "event": "steps"},
    ],
)
def test_webhook_ignores_events_it_cannot_act_on(client, webhook_dispatch, query_string):
    db_task = ProcessingTask(task_name="router_test")
    db_task.save(commit=True)

    resp = client.post(_path("WebhookView", db_id=db_task.id), query_string=query_string)

    assert resp.status_code == HTTPStatus.OK
    assert webhook_dispatch == []


def test_pipeline_options_are_offered_for_every_attribute(client):
    """Every option of the dropdown has to be accepted by the routing step."""
    db_task = ProcessingTask(task_name="router_test", parameters="{}")
    db_task.data["taxonomy_attributes"] = ["genre"]
    db_task.save(commit=True)

    body = client.get(_path("RoutingStepFrontend", db_id=db_task.id)).get_data(
        as_text=True
    )

    for key, label in PIPELINE_OPTIONS.items():
        assert f'value="{key}"' in body
        assert label in body
