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
from flask.views import MethodView
from flask_smorest import abort

from qhana_plugin_runner.api.plugin_schemas import (
    ProgressMetadata,
    ProgressMetadataSchema,
    StepMetadata,
    StepMetadataSchema,
)
from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.db.models.tasks import ProcessingTask, Step, TaskFile
from qhana_plugin_runner.storage import STORE

TASKS_API = SmorestBlueprint(
    "tasks-api",
    __name__,
    description="Api to request results of an async task.",
    url_prefix="/tasks",
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

    @ma.post_load
    def make_object(self, data, **kwargs):
        return OutputDataMetadata(**data)


@dataclass()
class TaskStatus:
    status: str
    log: Optional[str] = None
    progress: Optional[ProgressMetadata] = None
    steps: Sequence[StepMetadata] = field(default_factory=list)
    outputs: Sequence[OutputDataMetadata] = field(default_factory=list)


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

    @ma.post_dump()
    def remove_empty_attributes(self, data: Dict[str, Any], **kwargs):
        """Remove result attributes from serialized tasks that have not finished."""
        if data["log"] == None:
            del data["log"]
            del data["outputs"]
        if data["steps"] == None:
            del data["steps"]
        if data["progress"] == None:
            del data["progress"]
        return data

    @ma.post_load
    def make_object(self, data, **kwargs):
        return TaskStatus(**data)


@TASKS_API.route("/<int:task_id>/")
class TaskView(MethodView):
    """Task status resource."""

    @TASKS_API.response(HTTPStatus.OK, TaskStatusSchema())
    def get(self, task_id: int):
        """Get the current task status."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=task_id)
        if task_data is None:
            abort(HTTPStatus.NOT_FOUND, message="Task not found.")
            return  # return for type checker, abort raises exception

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
                        links=step.links,
                    )
                )

        if not task_data.is_finished:
            return TaskStatus(
                progress=progress,
                steps=steps,
                status=task_data.status,
                log=task_data.task_log,
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

        return TaskStatus(
            progress=progress,
            steps=steps,
            status=task_data.status,
            log=task_data.task_log,
            outputs=outputs,
        )

    # TODO add delete endpoint (and maybe serve result from different endpoint)
