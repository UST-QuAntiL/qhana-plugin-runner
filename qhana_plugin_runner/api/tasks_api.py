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
    ConcreteFileMetadataSchema,
    FileMetadata,
)
from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile
from qhana_plugin_runner.storage import STORE

TASKS_API = SmorestBlueprint(
    "tasks-api",
    __name__,
    description="Api to request results of an async task.",
    url_prefix="/tasks",
)


@dataclass()
class TaskData:
    name: str
    task_id: str
    status: str
    result: Optional[str] = None
    files: Sequence[FileMetadata] = field(default_factory=list)


class TaskStatusSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    status = ma.fields.String(required=True, allow_none=False, dump_only=True)
    result = ma.fields.String(required=False, allow_none=True, dump_only=True)
    files = ma.fields.List(
        ma.fields.Nested(ConcreteFileMetadataSchema()),
        required=False,
        allow_none=True,
        dump_only=True,
    )

    @ma.post_dump()
    def remove_empty_attributes(self, data: Dict[str, Any], **kwargs):
        """Remove result attributes from serialized tasks that have not finished."""
        if data["result"] == None:
            del data["result"]
            del data["files"]
        return data


@TASKS_API.route("/<string:task_id>/")
class TaskView(MethodView):
    """Task status resource."""

    @TASKS_API.response(HTTPStatus.OK, TaskStatusSchema())
    def get(self, task_id: str):
        """Get the current task status."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_task_id(
            task_id=task_id
        )
        if task_data is None:
            abort(HTTPStatus.NOT_FOUND, message="Task not found.")
            return  # return for type checker, abort raises exception

        assert (
            task_data.task_id is not None
        ), "If this is None then get_task_by_id is faulty."  # assertion for type checker

        if task_data.finished_at is None or task_data.finished_status is None:
            task_result = AsyncResult(task_id, app=CELERY)
            return TaskData(
                name=task_data.task_name,
                task_id=task_data.task_id,
                status=task_result.status,
                # TODO better result handling (store result in db, use db in this endpoint for finished tasks, ...)
                # TODO task result garbage collection (auto delete old (~7d default) results to free up resources again)
                result=None,  # only return a result if task is marked finished in db
            )

        files: List[FileMetadata] = []

        for file_ in TaskFile.get_task_result_files(task_data):
            if file_.file_type is None or file_.mimetype is None:
                continue  # result files must have file and mime type set
            files.append(
                FileMetadata(
                    file_type=file_.file_type,
                    mimetype=file_.mimetype,
                    filename=file_.file_name,
                    href=STORE.get_task_file_url(file_),
                )
            )

        return TaskData(
            name=task_data.task_name,
            task_id=task_data.task_id,
            status=task_data.finished_status,
            result=task_data.task_result,
            files=files,
        )

    # TODO add delete endpoint (and maybe serve result from different endpoint)
