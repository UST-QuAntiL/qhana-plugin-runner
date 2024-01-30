# Copyright 2024 QHAna plugin runner contributors.
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

"""Module containing helper functions for invoking other plugins from plugins."""

from typing import Any, Dict, Optional, Tuple, Literal
from urllib.parse import urljoin

from requests.exceptions import ConnectionError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.requests import REQUEST_SESSION, open_url


def get_plugin_endpoint(
    plugin_url: str, interaction_endpoint: Optional[str] = None
) -> str:
    with open_url(plugin_url) as metadata:
        decoded = metadata.json()
        url = decoded.get("entryPoint").get("href")
        if interaction_endpoint:
            raise NotImplementedError
        return urljoin(plugin_url, url)


def call_plugin_endpoint_and_get_monitor_task(endpoint_url, data: Dict[str, Any]) -> str:
    response = REQUEST_SESSION.post(
        endpoint_url,
        data=data,
        timeout=5,
        allow_redirects=True,
    )

    if not (200 <= response.status_code < 300):
        print(response.text)  # TODO better debug methods...

    response.raise_for_status()

    result_url = response.url
    response.close()

    return result_url


def get_task_result_no_wait(
    result_url: str,
) -> Tuple[Literal["PENDING", "SUCCESS", "FAILURE"], Any]:
    with open_url(result_url, timeout=3) as result:
        result_data = result.json()
        status = result_data.get("status")
        if status in ("PENDING", "SUCCESS", "FAILURE"):
            return status, result_data
        else:
            return "FAILURE", result_data


class ResultUnchangedError(Exception):
    pass


@CELERY.task(
    name=f"{__name__}.monitor_result",
    bind=True,
    ignore_result=True,
    autoretry_for=(ResultUnchangedError, ConnectionError),
    retry_backoff=True,
    max_retries=None,
)
def monitor_result(
    self,
    result_url: str,
    webhook_url: str,
    monitor: Literal["status", "substeps", "all"] = "all",
) -> None:
    status, result = get_task_result_no_wait(result_url)

    if status in ("SUCCESS", "FAILURE"):
        if monitor == "status" or monitor == "all":
            # status has changed, monitor webhook
            REQUEST_SESSION.post(webhook_url, timeout=1)
        return  # result cannot change further

    if monitor == "substeps" or monitor == "all":
        steps = result.get("steps", [])
        current_step = steps[-1] if steps else None
        if current_step and not current_step.get("cleared", False):
            # found uncleared step
            REQUEST_SESSION.post(webhook_url, timeout=1)
            return  # stop monitoring to not waste power on waiting for human inputs

    raise ResultUnchangedError  # retry later
