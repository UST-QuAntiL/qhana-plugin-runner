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
from urllib.parse import urlsplit

from flask import url_for

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from router import ROUTER_BLP, Router
from router.schemas import WU_PALMER_PLUGIN, MAPPING_PLUGIN, PIPELINE_OPTIONS

from tests.utils import mock_task_dispatch


def _path(endpoint: str, **kwargs) -> str:
    return urlsplit(url_for(f"{ROUTER_BLP.name}.{endpoint}", **kwargs)).path


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
