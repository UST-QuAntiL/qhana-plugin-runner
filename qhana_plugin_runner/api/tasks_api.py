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
from celery.result import AsyncResult
from flask.views import MethodView
from flask_smorest import abort

from qhana_plugin_runner.api.plugin_schemas import (
    ProgressMetadata,
    ProgressMetadataSchema,
    StepMetadata,
    StepMetadataSchema,
    DataMetadata,
    DataMetadataSchema,
)
from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask, Step, TaskFile

TASKS_API = SmorestBlueprint(
    "tasks-api",
    __name__,
    description="Api to request results of an async task.",
    url_prefix="/tasks",
)


@dataclass()
class TaskData:
    status: str
    log: Optional[str] = None
    progress: ProgressMetadata = None
    steps: Sequence[StepMetadata] = field(default_factory=list)
    outputs: Sequence[DataMetadata] = field(default_factory=list)


class TaskStatusSchema(MaBaseSchema):
    status = ma.fields.String(required=True, allow_none=False, dump_only=True)
    log = ma.fields.String(required=False, allow_none=True, dump_only=True)
    progress = ma.fields.Nested(
        ProgressMetadataSchema, required=False, allow_none=True, dump_only=True
    )
    steps = ma.fields.List(
        ma.fields.Nested(StepMetadataSchema),
        required=False,
        allow_none=True,
        dump_only=True,
    )
    outputs = ma.fields.List(
        ma.fields.Nested(DataMetadataSchema()),
        required=False,
        allow_none=True,
        dump_only=True,
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


@TASKS_API.route("/<string:task_id>/")
class TaskView(MethodView):
    """Task status resource."""

    @TASKS_API.response(HTTPStatus.OK, TaskStatusSchema())
    def get(self, task_id: str):
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

        steps = None
        if len(task_data.steps) > 0:
            steps: List[StepMetadata] = []
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

        if not task_data.is_finished:
            return TaskData(
                progress=progress,
                steps=steps,
                status=task_data.status,
                log=task_data.task_log,
            )

        outputs: List[DataMetadata] = []

        for file_ in TaskFile.get_task_result_files(task_data):
            if file_.file_type is None or file_.mimetype is None:
                continue  # result files must have file and mime type set
            outputs.append(
                DataMetadata(
                    data_type=file_.file_type,
                    content_type=[file_.mimetype],
                    required=True,  # TODO: use correct value
                )
            )

        return TaskData(
            progress=progress,
            steps=steps,
            status=task_data.status,
            log=task_data.task_log,
            outputs=outputs,
        )

    # TODO add delete endpoint (and maybe serve result from different endpoint)
