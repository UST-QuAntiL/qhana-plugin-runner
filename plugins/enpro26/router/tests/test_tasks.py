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

import json

import pytest

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from router.tasks import handle_webhook_task, route_task

from tests.utils import MockResponse, run_task

pytestmark = pytest.mark.usefixtures("celery_worker")


def _setup_mock_task() -> ProcessingTask:
    params = {
        "entitiesUrl": "http://mock/entities",
        "entitiesMetadataUrl": "http://mock/meta",
        "taxonomiesZipUrl": "http://mock/tax",
        "transformer": "linear_inverse",
        "aggregator": "mean",
        "missingDataHandling": "ignore",
        "dimensions": 2,
        "metric": "metric_mds",
        "nInit": 4,
        "maxIter": 300,
    }
    db_task = ProcessingTask(task_name=route_task.name, parameters=json.dumps(params))
    db_task.data["webhook_url"] = "http://my-router/webhook"
    # The Wu-Palmer attribute list is computed in the routing step before the
    # pipeline tasks run.
    db_task.data["wu_palmer_attributes"] = "attr1"
    # The routing step resolves and stores the pipeline plugin metadata urls.
    db_task.data["plugin_urls"] = {
        "wu_palmer": "http://localhost:5005/plugins/wu-palmer/",
        "sym_max_mean": "http://localhost:5005/plugins/sym-max-mean/",
        "transformer": "http://localhost:5005/plugins/attr-sim-to-attr-dist-transformers/",
        "aggregator": "http://localhost:5005/plugins/distance-aggregator/",
        "mds": "http://localhost:5005/plugins/mds/",
    }
    db_task.save(commit=True)
    return db_task


def test_route_task_starts_wu_palmer(monkeypatch):
    db_task = _setup_mock_task()

    # MOCK HTTP CALLS
    def mock_post(url, **kwargs):
        if "wu-palmer" in url:
            # Mock the redirect location from Wu-Palmer creation
            return MockResponse(
                url,
                "text/html",
                status_code=303,
                headers={
                    "Location": "http://localhost:5005/plugins/wu-palmer@v0-2-1/tasks/999/"
                },
            )
        if "subscribe" in url:
            # Mock the subscription success
            return MockResponse(url, "application/json", status_code=200)
        raise ValueError(f"Unexpected POST to {url}")

    def mock_get(url, **kwargs):
        if "/tasks/999/" in url:
            # Mock the Wu-Palmer status check for subscription
            return MockResponse(
                url,
                "application/json",
                status_code=200,
                json_data={
                    "status": "PENDING",
                    "links": [{"type": "subscribe", "href": "http://mock/subscribe"}],
                },
            )
        raise ValueError(f"Unexpected GET to {url}")

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("requests.get", mock_get)
    # Resolve the process endpoint from the stored metadata url without HTTP.
    monkeypatch.setattr(
        "router.tasks.get_plugin_endpoint", lambda base: base + "process/"
    )

    # Execute Step 1
    run_task(route_task, db_id=db_task.id)

    # Verify State Machine
    db_task = ProcessingTask.get_by_id(db_task.id)
    assert (
        db_task.data["wu_palmer_url"]
        == "http://localhost:5005/plugins/wu-palmer@v0-2-1/tasks/999/"
    )


def test_handle_webhook_routes_to_step_2(monkeypatch):
    db_task = _setup_mock_task()
    wu_palmer_task_url = "http://mock/tasks/999/"
    db_task.data["wu_palmer_url"] = wu_palmer_task_url
    db_task.save(commit=True)

    # Mock the Webhook payload returning SUCCESS
    def mock_get(url, **kwargs):
        return MockResponse(
            url,
            "application/json",
            status_code=200,
            json_data={
                "status": "SUCCESS",
                "outputs": [
                    {
                        "dataType": "relation/element-similarities",
                        "href": "http://mock/sims.zip",
                    }
                ],
            },
        )

    # Mock the trigger to the next step so it doesn't actually run in this test
    triggered = []

    def mock_delay(*args, **kwargs):
        triggered.append("step_2_triggered")

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr("router.tasks.process_step_2_smm.delay", mock_delay)

    # Simulate webhook hitting the traffic cop
    run_task(handle_webhook_task, db_id=db_task.id, source_url=wu_palmer_task_url)

    # Ensure Traffic Cop routed correctly!
    assert len(triggered) == 1
    assert triggered[0] == "step_2_triggered"


def test_handle_webhook_ignores_unrecognized_source():
    db_task = _setup_mock_task()

    # Feed an unknown URL to the webhook
    result = run_task(
        handle_webhook_task, db_id=db_task.id, source_url="http://unknown-source"
    )

    assert result == "Unrecognized webhook"
