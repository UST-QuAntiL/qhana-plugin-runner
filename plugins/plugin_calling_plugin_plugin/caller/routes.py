# Copyright 2022 QHAna plugin runner contributors.
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
from typing import Mapping, Optional
from logging import Logger

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.plugin_utils.url_utils import get_plugin_name_from_plugin_url

from . import CALLER_BLP, Caller
from .schemas import (
    CallerTaskResponseSchema,
    CallerSelectCalleePluginSchema,
    CallerInputParametersSchema,
)
from .tasks import processing_task_1, processing_task_2
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result, add_step


@CALLER_BLP.route("/")
class MetadataView(MethodView):
    """Plugins collection resource."""

    @CALLER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @CALLER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Caller endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Caller plugin",
            description=Caller.instance.description,
            name=Caller.instance.name,
            version=Caller.instance.version,
            type=PluginType.processing,
            tags=Caller.instance.tags,
            entry_point=EntryPoint(
                href=url_for(
                    f"{CALLER_BLP.name}.ProcessStep1View"
                ),  # URL for the first process endpoint
                ui_href=url_for(
                    f"{CALLER_BLP.name}.MicroFrontendStep1"
                ),  # URL for the first micro frontend endpoint
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    ),
                ],
            ),
        )


@CALLER_BLP.route("/ui-step-1/")
class MicroFrontendStep1(MethodView):
    """Micro frontend for step 1 of the caller plugin."""

    example_inputs = {
        "inputStr": "opt test step 1",
    }

    @CALLER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for step 1 of the caller plugin.",
    )
    @CALLER_BLP.arguments(
        CallerSelectCalleePluginSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @CALLER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @CALLER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for step 1 of the caller plugin.",
    )
    @CALLER_BLP.arguments(
        CallerSelectCalleePluginSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @CALLER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = CallerSelectCalleePluginSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=Caller.instance.name,
                version=Caller.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{CALLER_BLP.name}.ProcessStep1View"
                ),  # URL of the first processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{CALLER_BLP.name}.MicroFrontendStep1",  # URL of this endpoint
                    **self.example_inputs,
                ),
            )
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@CALLER_BLP.route("/process-step-1/")
class ProcessStep1View(MethodView):
    """Start the processing task of step 1."""

    @CALLER_BLP.arguments(
        CallerSelectCalleePluginSchema(unknown=EXCLUDE), location="form"
    )
    @CALLER_BLP.response(HTTPStatus.OK, CallerTaskResponseSchema())
    @CALLER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name=processing_task_1.name,
        )
        db_task.save(commit=True)

        db_task.data["input_str"] = arguments["input_str"]
        db_task.data["invoked_plugin"] = arguments["callee_plugin_selector"]
        db_task.data["next_step_id"] = "processing-returned-data"
        db_task.data["href"] = url_for(
            f"{CALLER_BLP.name}.ProcessStep2View",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["ui_href"] = url_for(
            f"{CALLER_BLP.name}.MicroFrontendStep2",
            db_id=db_task.id,
            _external=True,
        )
        db_task.save(commit=True)

        # extract the plugin name from the plugin url
        plugin_url = arguments["callee_plugin_selector"]
        plugin_name = get_plugin_name_from_plugin_url(plugin_url)
        # add new step where the "callee" plugin is executed

        # name of the next step
        step_id = "invoked plugin"
        # URL of the process endpoint of the invoked plugin
        href = url_for(
            f"{plugin_name}.ProcessView", db_id=db_task.id, _external=True
        )  # FIXME replace the process view with the actual name of the first processing step of the invoked plugin
        # URL of the micro frontend endpoint of the invoked plugin
        ui_href = url_for(
            f"{plugin_name}.MicroFrontend", db_id=db_task.id, _external=True
        )  # FIXME replace the micro frontend with the actual name of the first ui step of the invoked plugin

        # Chain the first processing task with executing the "callee" plugin.
        # All tasks use the same db_id to be able to fetch data from the previous steps and to store data for the next
        # steps in the database.
        task: chain = processing_task_1.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=33
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@CALLER_BLP.route("/<int:db_id>/ui-step-2/")
class MicroFrontendStep2(MethodView):
    """Micro frontend for step 2 of the caller plugin."""

    example_inputs = {
        "inputStr": "opt test step 2",
    }

    @CALLER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for step 2 of the caller plugin.",
    )
    @CALLER_BLP.arguments(
        CallerInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @CALLER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @CALLER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for step 2 of the caller plugin.",
    )
    @CALLER_BLP.arguments(
        CallerInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @CALLER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if not data:
            # retrieve data from the invoked plugin
            input_str = db_task.data.get("input_str")
            data = {"inputStr": "Data from invoked plugin: " + input_str}

        schema = CallerInputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=Caller.instance.name,
                version=Caller.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{CALLER_BLP.name}.ProcessStep2View",
                    db_id=db_id,  # URL of the second processing step
                ),
                example_values=url_for(
                    f"{CALLER_BLP.name}.MicroFrontendStep2",  # URL of the second micro frontend
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@CALLER_BLP.route("/<int:db_id>/process-step-2/")
class ProcessStep2View(MethodView):
    """Start the processing task of step 2."""

    @CALLER_BLP.arguments(CallerInputParametersSchema(unknown=EXCLUDE), location="form")
    @CALLER_BLP.response(HTTPStatus.OK, CallerTaskResponseSchema())
    @CALLER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the demo task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.data["input_str"] = arguments["input_str"]
        db_task.clear_previous_step()
        db_task.save(commit=True)

        # Chain the second processing task with executing the task that saves the results and ends the execution.
        task: chain = processing_task_2.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
