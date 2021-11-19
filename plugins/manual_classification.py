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
from marshmallow.utils import INCLUDE

import requests
from plugins.costume_loader_pkg.schemas import InputParameters
import random
from http import HTTPStatus
from json import dumps, loads, JSONEncoder

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
    PluginMetadataSchema,
)
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
from qhana_plugin_runner.tasks import (
    add_step,
    save_step_error,
    save_task_error,
    save_task_result,
)
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from tempfile import SpooledTemporaryFile
from qhana_plugin_runner.storage import STORE
from flask import redirect

_plugin_name = "manual-classification"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

MANUAL_CLASSIFICATION_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Manual Classification API.",
    template_folder="manual_classification_templates",
)


class ResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class LoadParametersSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True,
        allow_none=False,
        load_only=True,
        metadata={"label": "Entities URL"},
    )


class ClassificationSchema(FrontendFormBaseSchema):
    class_identifier = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Class Name",
            "description": "Name of the class to be annotated",
            "input_type": "textfield",
        },
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


@MANUAL_CLASSIFICATION_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, ResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Entity filter endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Manual Classification",
            description="Manually annotate classes for data sets from MUSE database.",
            name=ManualClassification.instance.name,
            version=ManualClassification.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.ProcessView"),
                ui_href=url_for(f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontend"),
                data_input=[  # TODO: only file input (entities...)
                    DataMetadata(
                        data_type="raw",
                        content_type=[
                            "application/json",
                            "application/zip",
                        ],  # TODO: OR -> json, csv... scatch, not finalized yet
                        required=True,
                    )
                ],
                data_output=[  # TODO
                    DataMetadata(
                        data_type="raw", content_type=["application/json"], required=True
                    )
                ],
            ),
            tags=["data-annotation"],
        )


##########################
###### Data loading ######
##########################
@MANUAL_CLASSIFICATION_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the manual classification plugin."""

    example_inputs = {
        "inputFileUrl": "file:///<path_to_file>/entities.json",
    }

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the manual classification plugin."
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        LoadParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the manual classification plugin."
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        LoadParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = LoadParametersSchema()
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


@MANUAL_CLASSIFICATION_BLP.route("/load/")
class LoadView(MethodView):
    """Start a data preprocessing task."""

    @MANUAL_CLASSIFICATION_BLP.arguments(
        LoadParametersSchema(unknown=EXCLUDE), location="form"
    )
    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the data preprocessing task."""
        db_task = ProcessingTask(
            task_name="manual-classification", parameters=dumps(arguments)
        )
        db_task.save(commit=True)

        # add classification step
        step_id = "classification"
        href = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.ClassificationView", db_id=db_task.id
        )
        ui_href = url_for(
            f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontendClassification",
            db_id=db_task.id,
        )

        # all tasks need to know about db id to load the db entry
        task: chain = pre_render_classification.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=30
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", db_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@CELERY.task(
    name=f"{ManualClassification.instance.identifier}.pre_render_classification_task",
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
        task_data.data["mimetype"] = mimetype
        input_entities = ensure_dict(load_entities(file_=url_data, mimetype=mimetype))

        if not input_entities:
            msg = "No entities could be loaded!"
            TASK_LOGGER.error(msg)
            raise ValueError(msg)

        # extract relevant information to create selection fields for manual classification
        entity_classification_data = {}
        for entity in input_entities:
            # keys are entity id's and values are lists of annotated classes
            entity_classification_data[entity["ID"]] = []

    # store in data of task_data
    task_data.data["entity_data"] = dumps(entity_classification_data)
    task_data.data["input_file_url"] = input_file_url
    task_data.save(commit=True)

    return "Classification pre-rendering successful."


#################################
###### Classification step ######
#################################


@MANUAL_CLASSIFICATION_BLP.route("/<string:db_id>/classification-ui/")
class MicroFrontendClassification(MethodView):
    """Micro frontend for the classification step of the manual classification plugin."""

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the classification step of the manual classification plugin.",
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        ClassificationSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: str):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id)

    @MANUAL_CLASSIFICATION_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the classification step of the manual classification plugin.",
    )
    @MANUAL_CLASSIFICATION_BLP.arguments(
        ClassificationSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: str):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id)

    def render(self, data: Mapping, errors: dict, db_id: str):
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if task_data is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        task_data.clear_previous_step(commit=True)

        # retrive data to build frontend
        schema = ClassificationSchema()
        return Response(
            render_template(
                "manual_classification_template.html",
                name=ManualClassification.instance.name,
                version=ManualClassification.instance.version,
                schema=schema,
                values=data,
                id_list=loads(task_data.data["entity_data"]).keys(),
                errors=errors,
                process=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.ClassificationView", db_id=db_id
                ),
                done=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.ClassificationDoneView",
                    db_id=db_id,
                ),
                example_values=url_for(
                    f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontendClassification",
                    db_id=db_id,
                ),
            )
        )  # TODO: should have two process buttons, one Continue (add_another) and Done (done) => add another creates background task to add class annotation to entity_data, done adds and finishes plugin, no new step... continue should return the same view... not sure how the ui would do that -> another step??? redirect to current? view might be best...


@MANUAL_CLASSIFICATION_BLP.route("/<string:db_id>/add_class/")
class ClassificationView(MethodView):
    """Start a classification processing task."""

    @MANUAL_CLASSIFICATION_BLP.arguments(
        ClassificationSchema(unknown=INCLUDE),
        location="form",  # TODO: this should cause fields not in schema (id's) to be included... not sure if this works
    )
    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: str):
        """Start the classification task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = dumps(arguments)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = add_class.s(db_id=db_task.id)

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        # TODO: reload classification step, not sure if that's how ui is working...
        return redirect(
            url_for(
                f"{MANUAL_CLASSIFICATION_BLP.name}.MicroFrontendClassification",
                db_id=db_id,
            )
        )


@MANUAL_CLASSIFICATION_BLP.route("/<string:db_id>/done/")  # TODO
class ClassificationDoneView(MethodView):
    """Start a classification processing task."""

    @MANUAL_CLASSIFICATION_BLP.arguments(
        ClassificationSchema(unknown=INCLUDE), location="form"
    )
    @MANUAL_CLASSIFICATION_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MANUAL_CLASSIFICATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: str):
        """Start the classification task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = dumps(arguments)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = (
            add_class.s(db_id=db_task.id)
            | save_classification.s(db_id=db_task.id)
            | save_task_result.s(db_id=db_task.id)
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", db_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@CELERY.task(
    name=f"{ManualClassification.instance.identifier}.add_class",
    bind=True,
)
def add_class(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new add class task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: Dict = loads(task_data.parameters or "{}")
    class_identifier: Optional[str] = params.get("class_identifier", None)

    TASK_LOGGER.info(
        f"Loaded input parameters from db: class_identifier='{class_identifier}, params='{params}'"
    )

    if (
        class_identifier is None or not class_identifier
    ):  # should not happen because of form validation
        msg = "No class identified provided!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    entity_data = loads(task_data.data["entity_data"])
    for id in entity_data.keys():
        if params.get(id):
            tmp = set(entity_data[id])
            tmp.add(class_identifier)
            entity_data[id] = list(tmp)

    # store in data of task_data
    task_data.data["entity_data"] = dumps(entity_data)
    task_data.save(commit=True)

    return "Adding new class successful."


@CELERY.task(
    name=f"{ManualClassification.instance.identifier}.save_classification",
    bind=True,
)
def save_classification(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new add class task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    try:
        input_file_url = task_data.data["input_file_url"]
    except:
        msg = f"No input_file_url in db! Somehow got lost during the plugin execution"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # TODO: somehow return output files with annotated data...
    annotated_entities = []
    for id in task_data.data["entity_data"].keys():
        annotated_entities.append(
            {"ID": id, "annotated_classes": task_data.data["entity_data"][id]}
        )

    with SpooledTemporaryFile(mode="w") as output:
        try:
            mimetype = task_data.data["mimetype"]
        except:
            mimetype = "application/json"
        save_entities(entities=annotated_entities, file_=output, mimetype=mimetype)

        if mimetype == "application/json":
            file_type = ".json"
        else:
            file_type = ".csv"
        STORE.persist_task_result(
            db_id,
            output,
            "filtered_entities" + file_type,
            "entity_filter_output",
            mimetype,
        )

    return "Adding new class successful."
