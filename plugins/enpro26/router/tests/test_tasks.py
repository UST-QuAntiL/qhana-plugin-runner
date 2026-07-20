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
        "dimensions": 2,
        "metric": "metric_mds",
        "nInit": 4,
        "maxIter": 300,
        "missingDataHandling": "mean",
    }
    db_task = ProcessingTask(task_name=route_task.name, parameters=json.dumps(params))
    db_task.data["webhook_url"] = "http://my-router/webhook"
    # The Wu-Palmer attribute list is computed in the routing step before the
    # pipeline tasks run.
    db_task.data["wu_palmer_attributes"] = "attr1"
    # The routing step resolves and stores the pipeline plugin metadata urls.
    db_task.data["plugin_urls"] = {
        "wu_palmer": "http://localhost:5005/plugins/wu-palmer/",
        "transformer": "http://localhost:5005/plugins/element_sim-to-element_dist-transformers/",
        "aggregator": "http://localhost:5005/plugins/attribute-distance-aggregator/",
        "mds": "http://localhost:5005/plugins/attribute-distance-mds/",
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
            return MockResponse(url, "application/zip", status_code=200)
        raise ValueError(f"Unexpected POST to {url}")

    def mock_get(url, **kwargs):
        if "/tasks/999/" in url:
            # Mock the Wu-Palmer status check for subscription
            return MockResponse(
                url,
                "application/zip",
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


@pytest.mark.parametrize(
    "source_key, mock_url, next_task_path",
    [
        (
            "wu_palmer_url",
            "http://mock/tasks/wp/999/",
            "router.tasks.process_step_2_transformers.apply_async",
        ),
        (
            "transformers_url",
            "http://mock/tasks/tr/999/",
            "router.tasks.process_step_3_aggregator.apply_async",
        ),
        (
            "aggregators_url",
            "http://mock/tasks/ag/999/",
            "router.tasks.process_step_4_mds.apply_async",
        ),
        (
            "mds_url",
            "http://mock/tasks/mds/999/",
            "router.tasks.finalize_pipeline.apply_async",
        ),
    ],
)
def test_handle_webhook_routing_all_steps(
    monkeypatch, source_key, mock_url, next_task_path
):
    """
    Tests that the webhook traffic cop successfully routes a SUCCESS status
    from a given step to the correct subsequent task using apply_async.
    """
    db_task = _setup_mock_task()
    db_task.data[source_key] = mock_url
    db_task.save(commit=True)

    # Mock the Webhook payload returning SUCCESS
    def mock_get(url, **kwargs):
        return MockResponse(
            url,
            "application/zip",
            status_code=200,
            json_data={
                "status": "SUCCESS",
                "outputs": [
                    {
                        "dataType": "some/datatype",
                        "href": "http://mock/data.zip",
                    }
                ],
            },
        )

    # Mock the trigger to the next step so it doesn't actually run in this test
    triggered = []

    def mock_apply_async(*args, **kwargs):
        # Assert that countdown=4 is being passed
        if kwargs.get("countdown") == 4:
            triggered.append("next_step_triggered")

    monkeypatch.setattr("requests.get", mock_get)
    monkeypatch.setattr(next_task_path, mock_apply_async)

    # Simulate webhook hitting the traffic cop
    run_task(handle_webhook_task, db_id=db_task.id, source_url=mock_url)

    assert len(triggered) == 1
    assert triggered[0] == "next_step_triggered"


def test_handle_webhook_ignores_unrecognized_source():
    db_task = _setup_mock_task()

    # Feed an unknown URL to the webhook
    result = run_task(
        handle_webhook_task, db_id=db_task.id, source_url="http://unknown-source"
    )

    assert result == "Unrecognized webhook"
