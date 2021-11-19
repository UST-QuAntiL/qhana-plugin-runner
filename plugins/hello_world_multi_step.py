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

from http import HTTPStatus
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional

import marshmallow as ma
from celery.canvas import chain
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from sqlalchemy.sql.expression import select

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
    PluginMetadataSchema,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import (
    add_step,
    save_step_error,
    save_task_error,
    save_task_result,
)
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "hello-world-multi-step"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


HELLO_MULTI_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Demo plugin API.",
    template_folder="hello_world_templates",
)


class HelloWorld(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return HELLO_MULTI_BLP


TASK_LOGGER = get_task_logger(__name__)


class DemoResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class HelloWorldParametersSchema(FrontendFormBaseSchema):
    input_str = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Input String",
            "description": "A simple string input.",
            "input_type": "textarea",
        },
    )


@HELLO_MULTI_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @HELLO_MULTI_BLP.response(HTTPStatus.OK, DemoResponseSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Demo endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Hello World Multi-Step Plugin",
            description="Simple multi-step plugin.",
            name=HelloWorld.instance.name,
            version=HelloWorld.instance.version,
            type=PluginType.complex,
        )


@HELLO_MULTI_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the hello world plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @HELLO_MULTI_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @HELLO_MULTI_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = HelloWorldParametersSchema()
        return Response(
            render_template(
                "hello_template.html",
                name=HelloWorld.instance.name,
                version=HelloWorld.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{HELLO_MULTI_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{HELLO_MULTI_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@HELLO_MULTI_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(unknown=EXCLUDE), location="form"
    )
    @HELLO_MULTI_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name="hello-world-multi-step-plugin", parameters=dumps(arguments)
        )  # TODO: change ProcessingTask class name and task_name???
        db_task.save(commit=True)

        # set previous step to cleared
        # nothing to do (no previous step)

        # next step
        step_id = "step1"
        href = url_for(f"{HELLO_MULTI_BLP.name}.Step1Frontend", db_id=db_task.id)
        ui_href = url_for(f"{HELLO_MULTI_BLP.name}.Step1View", db_id=db_task.id)

        # all tasks need to know about db id to load the db entry
        task: chain = preprocessing_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=50
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        # note that task_id has to be updated in each step
        db_task.task_id = result.id
        db_task.save(commit=True)

        # TODO: change multi-step ui to include progress data
        return redirect(
            url_for("tasks-api.TaskView", db_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@CELERY.task(name=f"{HelloWorld.instance.identifier}.preprocessing_task", bind=True)
def preprocessing_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_str: Optional[str] = loads(task_data.parameters or "{}").get("input_str", None)
    TASK_LOGGER.info(f"Loaded input parameters from db: input_str='{input_str}'")
    if input_str is None:
        raise ValueError("No input argument provided!")

    TASK_LOGGER.info("Some long running preprocessing...")
    task_data.data = {"x": "x1", "y": "y1"}
    task_data.data["input_str"] = input_str

    task_data.save(commit=True)

    if input_str:
        out_str = input_str.replace("input", "output")
        with SpooledTemporaryFile(mode="w") as output:
            output.write(out_str)
            STORE.persist_task_result(
                db_id, output, "out.txt", "hello-world-output", "text/plain"
            )
        return "result: " + repr(out_str)
    return "Empty input string, no output could be generated!"


@HELLO_MULTI_BLP.route("/<string:db_id>/step1-ui/")
class Step1Frontend(MethodView):
    """Micro frontend for the hello world plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @HELLO_MULTI_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: str):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @HELLO_MULTI_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: str):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: str, errors: dict):
        # TODO: retrieve and display data
        schema = HelloWorldParametersSchema()
        return Response(
            render_template(
                "hello_template.html",
                name=HelloWorld.instance.name,
                version=HelloWorld.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{HELLO_MULTI_BLP.name}.Step1View", db_id=db_id),
                example_values=url_for(
                    f"{HELLO_MULTI_BLP.name}.Step1Frontend",
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@HELLO_MULTI_BLP.route("/<string:db_id>/step1/")
class Step1View(MethodView):
    """Start a long running processing task."""

    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(unknown=EXCLUDE), location="form"
    )
    @HELLO_MULTI_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, error, db_id: str):
        """Start the demo task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

        # set previous step to cleared
        db_task.clear_previous_step(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = preprocessing_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", db_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


@CELERY.task(name=f"{HelloWorld.instance.identifier}.processing_task", bind=True)
def processing_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    TASK_LOGGER.info(
        "Main long running processing... Retrieving previously written values:"
    )
    TASK_LOGGER.info(task_data.data)
    TASK_LOGGER.info("input_str=" + task_data.data["input_str"])

    if task_data.data["input_str"]:
        out_str = task_data.data["input_str"].replace("input", "output")
        with SpooledTemporaryFile(mode="w") as output:
            output.write(out_str)
            STORE.persist_task_result(
                db_id, output, "out.txt", "hello-world-output", "text/plain"
            )
        return "result: " + repr(out_str)
    return "Empty input string, no output could be generated!"
