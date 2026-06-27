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

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from router.tasks import handle_webhook_task, route_task


class _MockResponse:
    def __init__(self, status_code: int, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = headers or {}
        self.text = "Mock Error"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP Error")


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
    db_task.save(commit=True)
    return db_task


def test_route_task_starts_wu_palmer(app, monkeypatch):
    db_task = _setup_mock_task()

    # MOCK HTTP CALLS
    def mock_post(url, **kwargs):
        if "wu-palmer" in url:
            # Mock the redirect location from Wu-Palmer creation
            return _MockResponse(
                303,
                headers={
                    "Location": "http://localhost:5005/plugins/wu-palmer@v0-2-1/tasks/999/"
                },
            )
        if "subscribe" in url:
            # Mock the subscription success
            return _MockResponse(200)
        raise ValueError(f"Unexpected POST to {url}")

    def mock_get(url, **kwargs):
        if "/tasks/999/" in url:
            # Mock the Wu-Palmer status check for subscription
            return _MockResponse(
                200,
                json_data={
                    "status": "PENDING",
                    "links": [{"type": "subscribe", "href": "http://mock/subscribe"}],
                },
            )
        raise ValueError(f"Unexpected GET to {url}")

    monkeypatch.setattr("requests.post", mock_post)
    monkeypatch.setattr("requests.get", mock_get)

    # Execute Step 1
    route_task.apply(kwargs={"db_id": db_task.id}).get()

    DB.session.expire_all()

    # Verify State Machine
    db_task = ProcessingTask.get_by_id(db_task.id)
    assert (
        db_task.data["wu_palmer_url"]
        == "http://localhost:5005/plugins/wu-palmer@v0-2-1/tasks/999/"
    )


def test_handle_webhook_routes_to_step_2(app, monkeypatch):
    db_task = _setup_mock_task()
    wu_palmer_task_url = "http://mock/tasks/999/"
    db_task.data["wu_palmer_url"] = wu_palmer_task_url
    db_task.save(commit=True)

    # Mock the Webhook payload returning SUCCESS
    def mock_get(url, **kwargs):
        return _MockResponse(
            200,
            json_data={
                "status": "SUCCESS",
                "outputs": [
                    {
                        "dataType": "custom/element-similarities",
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
    handle_webhook_task.apply(
        kwargs={"db_id": db_task.id, "source_url": wu_palmer_task_url}
    ).get()

    # Ensure Traffic Cop routed correctly!
    assert len(triggered) == 1
    assert triggered[0] == "step_2_triggered"


def test_handle_webhook_ignores_unrecognized_source(app, monkeypatch):
    db_task = _setup_mock_task()

    # Feed an unknown URL to the webhook
    result = handle_webhook_task.apply(
        kwargs={"db_id": db_task.id, "source_url": "http://unknown-source"}
    ).get()

    assert result == "Unrecognized webhook"
