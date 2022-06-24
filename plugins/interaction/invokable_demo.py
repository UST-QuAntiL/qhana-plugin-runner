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
from celery.utils.log import get_task_logger
from flask import Response, redirect, abort
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result, add_step
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "invokable-demo"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


INVOKABLE_DEMO_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Demo plugin that can be invoked by other plugins API.",
    template_folder="hello_world_templates",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParametersSchema(FrontendFormBaseSchema):
    input_str = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Input String",
            "description": "A simple string input.",
            "input_type": "textarea",
        },
    )


@INVOKABLE_DEMO_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @INVOKABLE_DEMO_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @INVOKABLE_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=InvokableDemo.instance.name,
            description=INVOKABLE_DEMO_BLP.description,
            name=InvokableDemo.instance.identifier,
            version=InvokableDemo.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(
                    f"{INVOKABLE_DEMO_BLP.name}.ProcessView", db_id=0
                ),  # URL for the first process endpoint  # FIXME: db_id
                ui_href=url_for(
                    f"{INVOKABLE_DEMO_BLP.name}.MicroFrontend", db_id=0
                ),  # URL for the first micro frontend endpoint  # FIXME: db_id
                plugin_dependencies=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )
                ],
            ),
            tags=[],
        )


@INVOKABLE_DEMO_BLP.route("/<int:db_id>/ui/")
class MicroFrontend(MethodView):
    """Micro frontend of the invokable demo plugin."""

    example_inputs = {
        "inputStr": "test2",
    }

    @INVOKABLE_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the invokable demo plugin."
    )
    @INVOKABLE_DEMO_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @INVOKABLE_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @INVOKABLE_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the invokable demo plugin."
    )
    @INVOKABLE_DEMO_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @INVOKABLE_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        plugin = InvokableDemo.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = InputParametersSchema()

        if not data:
            data = {
                "inputStr": "Data from invoking plugin: " + db_task.data.get("input_str")
            }

        return Response(
            render_template(
                "simple_template.html",
                name=InvokableDemo.instance.name,
                version=InvokableDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{INVOKABLE_DEMO_BLP.name}.ProcessView", db_id=db_id
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{INVOKABLE_DEMO_BLP.name}.MicroFrontend",
                    db_id=db_id,
                    **self.example_inputs,
                ),  # URL of this endpoint
            )
        )


@INVOKABLE_DEMO_BLP.route("/<int:db_id>/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @INVOKABLE_DEMO_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @INVOKABLE_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @INVOKABLE_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the invoked task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.data["input_str"] = arguments["input_str"]
        db_task.clear_previous_step()
        db_task.save(commit=True)

        # add the next processing step with the data that was stored by the previous step of the invoking plugin
        task: chain = demo_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=db_task.data["next_step_id"],  # name of the next sub-step
            href=db_task.data["href"],  # URL to the processing endpoint of the next step
            ui_href=db_task.data["ui_href"],  # URL to the micro frontend of the next step
            prog_value=66,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class InvokableDemo(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return INVOKABLE_DEMO_BLP


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{InvokableDemo.instance.identifier}.demo_task", bind=True)
def demo_task(self, db_id: int) -> str:
    """
    Demo processing task. Retrieves the input data from the database and stores it in a file.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting new invoked task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_str: str = task_data.data.get("input_str")
    TASK_LOGGER.info(f"Loaded input data from db: input_str='{input_str}'")

    if input_str is None:
        raise ValueError("No input data provided!")

    out_str = "User input from invoked plugin micro frontend: " + input_str
    with SpooledTemporaryFile(mode="w") as output:
        output.write(out_str)
        STORE.persist_task_result(
            db_id,
            output,
            "output_invoked_demo.txt",
            "hello-world-output",
            "text/plain",
        )
    return "result: " + repr(out_str)
