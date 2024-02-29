# Copyright 2021 QHAna plugin runner contributors.
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

"""Module containing endpoints related to task progress and results."""

from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Sequence

import marshmallow as ma
from flask import url_for
from flask.views import MethodView
from flask_smorest import abort
from marshmallow.validate import OneOf

from qhana_plugin_runner.api.plugin_schemas import (
    ProgressMetadata,
    ProgressMetadataSchema,
    StepMetadata,
    StepMetadataSchema,
)
from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import (
    ProcessingTask,
    Step,
    TaskFile,
    TaskUpdateSubscription,
)
from qhana_plugin_runner.storage import STORE

TASKS_API = SmorestBlueprint(
    "tasks-api",
    __name__,
    description="Api to request results of an async task.",
    url_prefix="/tasks",
)


class SubscriptionDataSchema(MaBaseSchema):
    command = ma.fields.String(
        required=True,
        allow_none=True,
        validate=OneOf(("subscribe", "unsubscribe")),
        metadata={"description": "Whether to subscribe or unsubscribe."},
    )
    event = ma.fields.String(
        required=False,
        allow_none=True,
        missing=None,
        metadata={"description": "The type of event to subscribe to."},
    )
    webhook_href = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={"description": "The URL of the wbhook subscribing to these events."},
    )


@dataclass
class OutputDataMetadata:
    data_type: str
    content_type: str
    href: str
    name: Optional[str] = None


class OutputDataMetadataSchema(MaBaseSchema):
    data_type = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={"description": "The type of the output data (e.g. distance-matrix)."},
    )
    content_type = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "description": "The media type (mime type) of the output data (e.g. application/json)."
        },
    )
    href = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={"description": "The URL of the output data."},
    )
    name = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={"description": "An optional human readable name for the output data."},
    )

    @ma.post_dump()
    def remove_empty_attributes(self, data: Dict[str, Any], **kwargs):
        """Remove name if it is none."""
        if data["name"] == None:
            del data["name"]
        return data


@dataclass()
class TaskLink:
    href: str
    type: str


class TaskLinkSchema(MaBaseSchema):
    type = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={"description": "The type of the link."},
    )
    href = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={"description": "The URL of the link."},
    )


@dataclass()
class TaskData:
    status: str
    log: Optional[str] = None
    progress: Optional[ProgressMetadata] = None
    steps: Sequence[StepMetadata] = field(default_factory=list)
    outputs: Sequence[OutputDataMetadata] = field(default_factory=list)
    links: Sequence[TaskLink] = field(default_factory=list)


class TaskStatusSchema(MaBaseSchema):
    status = ma.fields.String(required=True, allow_none=False)
    log = ma.fields.String(required=False, allow_none=True)
    progress = ma.fields.Nested(ProgressMetadataSchema, required=False, allow_none=True)
    steps = ma.fields.List(
        ma.fields.Nested(StepMetadataSchema),
        required=False,
        allow_none=True,
    )
    outputs = ma.fields.List(
        ma.fields.Nested(OutputDataMetadataSchema()),
        required=False,
        allow_none=True,
    )
    links = ma.fields.List(
        ma.fields.Nested(TaskLinkSchema()),
        required=False,
        allow_none=True,
    )

    @ma.post_dump()
    def remove_empty_attributes(self, data: Dict[str, Any], **kwargs):
        """Remove result attributes from serialized tasks that have not finished."""
        if data["log"] is None:
            del data["log"]
            del data["outputs"]
        if data["steps"] is None:
            del data["steps"]
        if data["progress"] is None:
            del data["progress"]
        return data

    @ma.post_load()
    def make_task_data(self, data: Dict[str, Any], **kwargs):
        return TaskData(**data)


@TASKS_API.route("/<int:task_id>/")
class TaskView(MethodView):
    """Task status resource."""

    @TASKS_API.response(HTTPStatus.OK, TaskStatusSchema())
    def get(self, task_id: int):
        """Get the current task status."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=task_id)
        if task_data is None:
            abort(HTTPStatus.NOT_FOUND, message="Task not found.")

        return self.convert_task_data(task_data)

    def convert_task_data(self, task_data: ProcessingTask):
        progress = None
        if task_data.progress_value:
            progress = {
                "value": task_data.progress_value,
                "start": task_data.progress_start,
                "target": task_data.progress_target,
                "unit": task_data.progress_unit,
            }

        steps: Optional[List[StepMetadata]] = None
        if len(task_data.steps) > 0:
            steps = []
            step: Step
            for step in task_data.steps:
                steps.append(
                    StepMetadata(
                        href=step.href,
                        uiHref=step.ui_href,
                        stepId=step.step_id,
                        cleared=step.cleared,
                    )
                )

        links = [
            TaskLink(
                type="subscribe",
                href=url_for(
                    "tasks-api.TaskView", task_id=str(task_data.id), _external=True
                ),
            )
        ]

        links += [TaskLink(href=link.href, type=link.type) for link in task_data.links]

        if not task_data.is_finished:
            return TaskData(
                progress=progress,
                steps=steps,
                status=task_data.status,
                log=task_data.task_log,
                links=links,
            )

        outputs: List[OutputDataMetadata] = []

        for file_ in TaskFile.get_task_result_files(task_data):
            if file_.file_type is None or file_.mimetype is None:
                continue  # result files must have file and mime type set
            href = STORE[file_.storage_provider].get_task_file_url(file_)
            outputs.append(
                OutputDataMetadata(
                    data_type=file_.file_type,
                    content_type=file_.mimetype,
                    href=href,
                    name=file_.file_name,
                )
            )

        return TaskData(
            progress=progress,
            steps=steps,
            status=task_data.status,
            log=task_data.task_log,
            outputs=outputs,
            links=links,
        )

    @TASKS_API.arguments(SubscriptionDataSchema(), location="json")
    @TASKS_API.response(HTTPStatus.OK, TaskStatusSchema())
    def post(self, command, task_id: int):
        """Subscribe to future task status updates."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=task_id)
        if task_data is None:
            abort(HTTPStatus.NOT_FOUND, message="Task not found.")

        match command["command"]:
            case "subscribe":
                self.subscribe(task_data, command)
            case "unsubscribe":
                self.unsubscribe(task_data, command)
            case cmd:
                abort(
                    HTTPStatus.BAD_REQUEST, message=f"Command '{cmd}' is not supported!"
                )

        return self.convert_task_data(task_data)

    def subscribe(self, task_data: ProcessingTask, subscription_data):
        subscriptions = TaskUpdateSubscription.get_by_task_and_subscriber(
            task=task_data,
            webhook_href=subscription_data["webhook_href"],
            event=subscription_data.get("event", None),
        )
        if subscriptions:
            return  # prevent multiple subscriptions for the same event type and webhook
        subscription = TaskUpdateSubscription(
            task=task_data,
            webhook_href=subscription_data["webhook_href"],
            task_href=url_for(
                f"{TASKS_API.name}.{TaskView.__name__}",
                task_id=task_data.id,
                _external=True,
            ),
            event_type=subscription_data.get("event"),
        )
        subscription.save(commit=True)

    def unsubscribe(self, task_data: ProcessingTask, subscription_data):
        subscriptions = TaskUpdateSubscription.get_by_task_and_subscriber(
            task=task_data,
            webhook_href=subscription_data["webhook_href"],
            event=subscription_data.get("event", ...),
        )
        for subscriber in subscriptions:
            DB.session.delete(subscriber)
        DB.session.commit()

    # TODO add delete endpoint (and maybe serve result from different endpoint)
