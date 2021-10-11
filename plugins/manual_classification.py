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

from enum import Enum
import mimetypes
from celery.app.task import Task

import requests
from plugins.costume_loader_pkg.schemas import InputParameters
import random
from http import HTTPStatus
from json import dumps, loads, JSONEncoder
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ResponseLike,
    ensure_dict,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.requests import open_url
from typing import Any, Dict, Generator, List, Mapping, Optional, Set, Union

import marshmallow as ma
from qhana_plugin_runner.api.extra_fields import CSVList, EnumField
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask_smorest import abort
from marshmallow import EXCLUDE
from sqlalchemy.sql.expression import select

from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from tempfile import SpooledTemporaryFile
from qhana_plugin_runner.storage import STORE
from flask import redirect

_plugin_name = "manual-classification"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

INFINITY = -1

MANUAL_CLASSIFICATION_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Manual Classification API.",
)


class ResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class WaitingSchema(MaBaseSchema):
    pass


class ManualClassificationParametersSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={"label": "Entities URL"},
    )


@MANUAL_CLASSIFICATION_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, ResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Entity filter endpoint returning the plugin metadata."""
        return {
            "name": ManualClassification.instance.name,
            "version": ManualClassification.instance.version,
            "identifier": ManualClassification.instance.identifier,
            "root_href": url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.PluginsView"),
            "title": "Entity loader",
            "description": "Filters data sets from the MUSE database.",
            "plugin_type": "data-loader",
            "tags": ["data:loading"],
            "processing_resource_metadata": {
                "href": url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.ProcessView"),
                "ui_href": url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontend"),
                "inputs": [  # TODO: only file input (entities...)
                    [
                        {
                            "output_type": "raw",
                            "content_type": "application/json",
                            "name": "Raw entity data",
                        },
                        {
                            "output_type": "raw",
                            "content_type": "text/csv",
                            "name": "Raw entity data",
                        },
                        # TODO: OR -> json, csv... scatch, not finalized yet
                    ]
                ],
                "outputs": [
                    [
                        {  # TODO: file handle to filtered file, could be json or csv...
                            "output_type": "raw",
                            "content_type": "application/json",
                            "name": "Filtered raw entity data",
                        },
                    ]
                ],
            },
        }


@MANUAL_CLASSIFICATION_BLP.route("/ui/", defaults={"task_id": None})
@MANUAL_CLASSIFICATION_BLP.route("/ui/<string:task_id>/")
class MicroFrontend(MethodView):
    """Micro frontend for the entity filter plugin."""

    example_inputs = {
        "inputFileUrl": "file:///<path_to_file>/entities.json",
    }

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the entity filter plugin."
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        ManualClassificationParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self, task_id: str, errors):
        """Return the micro frontend."""
        return self.render(request.args, task_id, errors)

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the entity filter plugin."
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        ManualClassificationParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, task_id: str, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, task_id, errors)

    def render(self, data: Mapping, task_id: str, errors: dict):
        if task_id is None:
            schema = ManualClassificationParametersSchema()
            return Response(
                render_template(
                    "simple_template.html",
                    name=ManualClassification.instance.name,
                    version=ManualClassification.instance.version,
                    schema=schema,
                    values=data,
                    errors=errors,
                    process=url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.LoadView"),
                    example_values=url_for(
                        f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontend",
                        **self.example_inputs,
                    ),
                )
            )
        else:
            return self.render_selection_step(data, task_id, errors)

    def render_selection_step(self, data: Mapping, task_id: str, errors: dict):
        """Get the current task status."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_task_id(
            task_id=task_id
        )
        if task_data is None:
            abort(HTTPStatus.NOT_FOUND, message="Task not found.")

        if not task_data.is_finished:
            # Case 1: task result not ready => some waiting view
            return Response(
                render_template(
                    "manual_classification_waiting.html",
                    name=ManualClassification.instance.name,
                    version=ManualClassification.instance.version,
                    schema=WaitingSchema(),
                )
            )

        else:
            # Case 2: task result ready => selection view
            self._render_selection_step(data, task_id, errors)

    def _render_selection_step(self, data: Mapping, task_id: str, errors: dict):
        # get
        pass


@MANUAL_CLASSIFICATION_BLP.route("/load/")
class LoadView(MethodView):
    """Start a long running processing task."""

    @MANUAL_CLASSIFICATION_BLP.arguments(
        ManualClassificationParametersSchema(unknown=EXCLUDE), location="form"
    )
    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, input_params: InputParameters):
        """Start the entity filter task."""
        db_task = ProcessingTask(
            task_name=pre_render_classification.name, parameters=dumps(input_params)
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = pre_render_classification.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.ui_base_endpoint_url = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontend"
        )
        db_task.ui_endpoint_url = url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.LoadView")
        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(result.id)), HTTPStatus.SEE_OTHER
        )


class ManualClassification(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return MANUAL_CLASSIFICATION_BLP

    def get_requirements(self) -> str:
        return ""


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{ManualClassification.instance.identifier}.manual_classification_task",
    bind=True,
)
def pre_render_classification(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new entity filter task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    input_file_url: Optional[str] = params.get("input_file_url", None)

    TASK_LOGGER.info(
        f"Loaded input parameters from db: input_file_url='{input_file_url}'"
    )

    if input_file_url is None or not input_file_url:
        msg = "No input file URL provided!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    input_entities = {}
    with open_url(input_file_url, stream=True) as url_data:
        try:
            mimetype = url_data.headers["Content-Type"]
        except KeyError:
            mimetype = mimetypes.MimeTypes().guess_type(url=input_file_url)[0]
        input_entities = ensure_dict(load_entities(file_=url_data, mimetype=mimetype))

    if not input_entities:
        msg = "No entities could be loaded!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    # TODO Step 1: extract relevant information to create selection fields for manual classification
    entity_classification_data = input_entities

    # TODO Step 2: store in data of task_data
    task_data.data["entity_data"] = entity_classification_data
    task_data.data["input_file_url"] = input_file_url
    task_data.save(commit=True)

    return "Classification pre-rendering successful."
