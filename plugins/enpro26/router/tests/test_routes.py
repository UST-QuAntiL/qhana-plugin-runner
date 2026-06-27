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


def _path(endpoint: str, **kwargs) -> str:
    return urlsplit(url_for(f"{ROUTER_BLP.name}.{endpoint}", **kwargs)).path


def test_metadata_endpoint_returns_full_descriptor(client):
    resp = client.get(url_for(f"{ROUTER_BLP.name}.PluginsView"))
    plugin = Router.instance

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_json()

    assert body["name"] == plugin.name
    assert body["type"] == "processing"
    assert len(body["entryPoint"]["dataOutput"]) == 5  # Ensures all 5 files are expected


def test_microfrontend_renders_form_fields(client):
    resp = client.get(url_for(f"{ROUTER_BLP.name}.MicroFrontend"))
    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)

    assert "Entities URL" in body
    assert "Routing Options" in body
    assert "Wu-Palmer" in body


def test_process_valid_payload_redirects_to_task(client):
    valid_payload = {
        "entitiesUrl": "http://example.com/data.csv",
        "entitiesMetadataUrl": "http://example.com/meta.json",
        "taxonomiesZipUrl": "http://example.com/tax.zip",
        "attributes": "instrumentation",
        "routingOptions": "wu_palmer",
        "transformer": "linear_inverse",
        "aggregator": "mean",
        "missingDataHandling": "ignore",
        "dimensions": 2,
        "metric": "metric_mds",
        "nInit": 4,
        "maxIter": 300,
    }

    resp = client.post(url_for(f"{ROUTER_BLP.name}.ProcessView"), data=valid_payload)

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert re.fullmatch(r"/tasks/\d+/", urlsplit(resp.headers["Location"]).path)


def test_webhook_view_accepts_status_events(client):
    # Create a dummy task in DB
    db_task = ProcessingTask(task_name="router_test")
    db_task.save(commit=True)

    resp = client.post(
        url_for(f"{ROUTER_BLP.name}.WebhookView", db_id=db_task.id),
        query_string={"source": "http://localhost/tasks/1/", "event": "status"},
    )

    assert resp.status_code == HTTPStatus.OK
    assert resp.get_data(as_text=True) == '"Webhook received"\n'
