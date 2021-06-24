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

# originally from <https://github.com/buehlefs/flask-template/>

"""Module containing the root endpoint of the v1 API."""

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, List, Optional

import marshmallow as ma
from celery.result import AsyncResult
from flask.helpers import url_for
from flask.views import MethodView
from flask_smorest import abort
from werkzeug.utils import redirect

from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.celery import CELERY

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
    result: Any


class TaskStatusSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    status = ma.fields.String(required=True, allow_none=False, dump_only=True)
    result = ma.fields.Raw(required=False, allow_none=True, dump_only=True)


@TASKS_API.route("/<string:task_id>/")
class TaskView(MethodView):
    """Task status resource."""

    @TASKS_API.response(HTTPStatus.OK, TaskStatusSchema())
    def get(self, task_id: str):
        """Get the current task status."""
        task_result = AsyncResult(task_id, app=CELERY)
        return TaskData(
            name=task_result.name,
            task_id=str(task_result.id),
            status=task_result.status,
            # TODO better result handling (store result in db, use db in this endpoint for finished tasks, ...)
            # TODO task result garbage collection (auto delete old (~7d default) results to free up resources again)
            result=task_result.result if task_result.successful() else None,
        )
