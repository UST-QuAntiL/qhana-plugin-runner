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

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
from urllib.parse import urljoin

from requests.exceptions import ConnectionError, RequestException

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.requests import REQUEST_SESSION, open_url


def get_plugin_endpoint(
    plugin_url: str, interaction_endpoint: Optional[str] = None
) -> str:
    """Retrieve a specific plugin endpoint url.

    Args:
        plugin_url (str): the base url of the plugin.
        interaction_endpoint (str, optional): if set, look for a link matching this type. Defaults to None (return the default processing endpoint).

    Returns:
        str: the default processing endpoint or the matching interaction endpoint
    """
    with open_url(plugin_url) as metadata:
        decoded = metadata.json()
        url = decoded.get("entryPoint").get("href")
        if interaction_endpoint:
            raise NotImplementedError
        return urljoin(plugin_url, url)


def call_plugin_endpoint(
    endpoint_url: str, data: Dict[str, Any], debug: bool = False
) -> str:
    """Call a plugin endpoint and return the result url.

    Only use this with the main entry point of a plugin.

    Args:
        endpoint_url (str): the main processing endpoint url of the plugin.
        data: (Dict[str, Any]): the data to pass to the endpoint.
        debug: (bool): if debg is True and the response status code indicates an error print the response body. Only use this during development. Defaults to False

    Returns:
        str: url of the processing task result
    """
    response = REQUEST_SESSION.post(
        endpoint_url,
        data=data,
        timeout=5,
        allow_redirects=True,
    )

    if debug and not (200 <= response.status_code < 300):
        print(response.text)  # TODO better debug methods...

    response.raise_for_status()

    result_url = response.url
    response.close()

    return result_url


def get_task_result_no_wait(
    result_url: str,
) -> Tuple[Literal["PENDING", "SUCCESS", "FAILURE"], Any]:
    """Get the task result with a minimal timeout.

    Args:
        result_url (str): the task result url to fetch.

    Returns:
        status, result: a tuple containing the normalized task status and the task result
    """
    with open_url(result_url, timeout=3) as result:
        result_data = result.json()
        status = result_data.get("status")
        if status in ("PENDING", "SUCCESS", "FAILURE"):
            return status, result_data
        else:
            return "FAILURE", result_data


class ResultUnchangedError(Exception):
    pass


def _check_result_for_updates(
    status: Literal["PENDING", "SUCCESS", "FAILURE"], result
) -> Optional[Literal["status", "steps"]]:
    if status in ("SUCCESS", "FAILURE"):
        return "status"

    steps = result.get("steps", [])
    current_step = steps[-1] if steps else None
    if current_step and not current_step.get("cleared", False):
        return "steps"
    return None  # no updates found


def _check_result_for_cleared_substep(
    status: Literal["PENDING", "SUCCESS", "FAILURE"], result, substep: Union[str, int]
) -> Optional[Literal["status", "steps"]]:
    if status in ("SUCCESS", "FAILURE"):
        return "status"

    steps: List[dict] = result.get("steps", [])

    step: Optional[dict] = None

    if isinstance(substep, int):
        if len(steps) > substep >= 0:
            step = steps[substep]
        else:
            raise ValueError("Substep out of bounds!")
    else:
        for s in reversed(steps):
            if s.get("stepId") == substep:
                step = s
        else:
            raise ValueError(f"Substep with ID {substep} not present!")

    if step is None:
        raise ValueError("Could not find step to monitor.")

    if step.get("cleared", False):
        return "steps"
    return None  # no updates found


def _subscribe_for_events(
    subscribe_url: str,
    webhook_url: str,
    events: Union[Literal["all"], Sequence[str]] = "all",
) -> Tuple[Literal["PENDING", "SUCCESS", "FAILURE"], Any]:
    result_data = None
    if events == "all":
        events = [None]
    for event in events:
        with REQUEST_SESSION.post(
            subscribe_url,
            json={"command": "subscribe", "webhookHref": webhook_url, "event": event},
            timeout=3,
        ) as result:
            result.raise_for_status()
            result_data = result.json()
    status = result_data.get("status") if result_data else "FAILURE"
    if status in ("PENDING", "SUCCESS", "FAILURE"):
        return status, result_data
    else:
        return "FAILURE", result_data


def subscribe(
    result_url: str,
    webhook_url: str,
    events: Union[Literal["all"], Sequence[str]] = "all",
    check_for_updates: bool = True,
) -> bool:
    """Subscribe to task result events.

    Args:
        result_url (str): the url of the task result.
        webhook_url (str): the webhook url that will receive event notifications.
        events ("all"|Sequence[str], optional): the type of events to subscribe to. Defaults to "all".
        check_for_updates (bool, optional): whether to check for (status and steps) updates after subscribing to handle possible race conditions. Defaults to True.

    Returns:
        bool: True if the subscription was registered
    """
    status, result = get_task_result_no_wait(result_url)

    subscribed = False
    if status == "PENDING" or True:
        for link in result.get("links", []):
            if link.get("type") == "subscribe":
                try:
                    status, result = _subscribe_for_events(
                        link["href"], webhook_url, events
                    )
                    subscribed = True
                except RequestException:
                    pass  # could not subscribe because of some error
                break

    if check_for_updates:
        updates = _check_result_for_updates(status, result)
        task = None
        if updates == "status" and (events == "all" or "status" in events):
            task = notify_webhook.s(webhook_url, result_url, "status")
        if updates == "substeps" and (events == "all" or "steps" in events):
            task = notify_webhook.s(webhook_url, result_url, "steps")

        if task:
            task.apply_async(delay=1)

    return subscribed


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
    monitor: Literal["status", "steps", "all"] = "all",
    retry: bool = True,
) -> None:
    """Monitor task result for status or step events by polling.

    Step events are limited to the finding of uncleared steps!

    Args:
        result_url (str): the task result url to monitor.
        webhook_url (str): the webhook to call on events.
        monitor ("status"|"steps"|"all", optional): Theevents to listen for. Defaults to "all".
        retry (bool, optional): whether the task should keep polling when no event was found. Defaults to True.
    """
    status, result = get_task_result_no_wait(result_url)

    updates = _check_result_for_updates(status, result)

    if updates and (updates == monitor or monitor == "all"):
        REQUEST_SESSION.post(
            webhook_url, params={"source": result_url, "event": updates}, timeout=1
        )
        return  # found an update, stop monitoring to save resources

    if retry:
        raise ResultUnchangedError  # retry later


@CELERY.task(
    name=f"{__name__}.monitor_external_substep",
    bind=True,
    ignore_result=True,
    autoretry_for=(ResultUnchangedError, ConnectionError),
    retry_backoff=True,
    max_retries=None,
)
def monitor_external_substep(
    self,
    result_url: str,
    webhook_url: str,
    substep: Union[str, int],
    retry: bool = True,
) -> None:
    """Monitor an external task for a specific step to be cleared.

    Args:
        result_url (str): the task result to monitor.
        webhook_url (str): the webhook to notify.
        substep (Union[str,int]): the str id of the step or the the interger index in the steps list of the step to monitor.
        retry (bool, optional): whether the task should keep polling when no event was found. Defaults to True.
    """
    status, result = get_task_result_no_wait(result_url)

    updates = _check_result_for_cleared_substep(status, result, substep)

    if updates:
        REQUEST_SESSION.post(
            webhook_url, params={"source": result_url, "event": updates}, timeout=1
        )
        return  # found an update, stop monitoring to save resources

    if retry:
        raise ResultUnchangedError  # retry later


@CELERY.task(
    name=f"{__name__}.notify_webhook",
    ignore_result=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=3,
)
def notify_webhook(
    webhook_url: str,
    result_url: str,
    event: str,
) -> None:
    REQUEST_SESSION.post(
        webhook_url, params={"source": result_url, "event": event}, timeout=1
    )
