# Copyright 2023 QHAna plugin runner contributors.
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

"""Module coontaining signal listeners."""

from typing import Optional

from flask import Flask

from .db.models.tasks import TaskUpdateSubscription
from .registry_client import PLUGIN_REGISTRY_CLIENT
from .tasks import (
    TASK_DETAILS_CHANGED,
    TASK_STATUS_CHANGED,
    TASK_STEPS_CHANGED,
    call_webhook,
)


def on_virtual_plugin_create(app, *, plugin_url, **extra):
    if not PLUGIN_REGISTRY_CLIENT.ready:
        return  # Cannot notify registry of this change
    PLUGIN_REGISTRY_CLIENT.fetch_by_rel(
        ["plugin", ["plugin", "post"]], query_params={"url": plugin_url}
    )


def on_virtual_plugin_remove(app, *, plugin_url, **extra):
    if not PLUGIN_REGISTRY_CLIENT.ready:
        return  # Cannot notify registry of this change
    PLUGIN_REGISTRY_CLIENT.fetch_by_rel(
        ["plugin", ["plugin", "post"]], query_params={"url": plugin_url}
    )


def on_task_update(app: Flask, *, task_id: int, event_type: Optional[str], **extra):
    subscribers = TaskUpdateSubscription.get_by_task_and_event(task_id, event_type)
    for webhook in subscribers:
        call_webhook.s(webhook_url=webhook.webhook_href, task_url=webhook.task_href, event_type=event_type).apply_async()


def on_task_status_update(app: Flask, *, task_id: int, **extra):
    on_task_update(app, task_id=task_id, event_type="status", **extra)


def on_task_steps_update(app: Flask, *, task_id: int, **extra):
    on_task_update(app, task_id=task_id, event_type="steps", **extra)


def on_task_details_update(app: Flask, *, task_id: int, **extra):
    on_task_update(app, task_id=task_id, event_type="details", **extra)


def register_signal_listeners(app: Flask):
    from .db.models.virtual_plugins import (
        VIRTUAL_PLUGIN_CREATED,
        VIRTUAL_PLUGIN_REMOVED,
    )

    VIRTUAL_PLUGIN_CREATED.connect(on_virtual_plugin_create, app)
    VIRTUAL_PLUGIN_REMOVED.connect(on_virtual_plugin_remove, app)

    TASK_STATUS_CHANGED.connect(on_task_status_update, app)
    TASK_STEPS_CHANGED.connect(on_task_steps_update, app)
    TASK_DETAILS_CHANGED.connect(on_task_details_update, app)
