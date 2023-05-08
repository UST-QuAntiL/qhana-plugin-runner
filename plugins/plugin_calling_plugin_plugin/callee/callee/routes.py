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

import os
from http import HTTPStatus
from typing import Mapping, Optional
from logging import Logger

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response
from flask import redirect, abort
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import CALLEE_BLP, Callee
from .schemas import CalleeInputParametersSchema, CalleeTaskResponseSchema
from .tasks import demo_task, demo_task_2
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result, add_step

TASK_LOGGER = get_task_logger(__name__)


@CALLEE_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @CALLEE_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @CALLEE_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=Callee.instance.name,
            description=Callee.instance.description,
            name=Callee.instance.identifier,
            version=Callee.instance.version,
            type=PluginType.processing,
            tags=["interaction", "callee", "invokable"],
            entry_point=EntryPoint(
                href=url_for(
                    f"{CALLEE_BLP.name}.ProcessView", db_id=0
                ),  # URL for the first process endpoint  # FIXME: db_id
                ui_href=url_for(
                    f"{CALLEE_BLP.name}.MicroFrontend", db_id=0
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
        )


@CALLEE_BLP.route("/<int:db_id>/ui-step-1/")
class MicroFrontend(MethodView):
    """Micro frontend of the callee plugin."""

    example_inputs = {
        "inputStr": "Callee plugin input string",
    }

    @CALLEE_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the callee plugin."
    )
    @CALLEE_BLP.arguments(
        CalleeInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @CALLEE_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @CALLEE_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the callee plugin."
    )
    @CALLEE_BLP.arguments(
        CalleeInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @CALLEE_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        plugin = Callee.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = CalleeInputParametersSchema()

        if not data:
            data = {
                "inputStr": "Data from invoking plugin: " + db_task.data.get("input_str")
            }

        return Response(
            render_template(
                "simple_template.html",
                name=Callee.instance.name,
                version=Callee.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{CALLEE_BLP.name}.ProcessView", db_id=db_id
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{CALLEE_BLP.name}.MicroFrontend",
                    db_id=db_id,
                    **self.example_inputs,
                ),  # URL of this endpoint
            )
        )


@CALLEE_BLP.route("/<int:db_id>/process-step-1/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @CALLEE_BLP.arguments(CalleeInputParametersSchema(unknown=EXCLUDE), location="form")
    @CALLEE_BLP.response(HTTPStatus.OK, CalleeTaskResponseSchema())
    @CALLEE_BLP.require_jwt("jwt", optional=True)
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

        href = url_for(
            f"{CALLEE_BLP.name}.MicroFrontendStep2", db_id=db_id, _external=True
        )

        ui_href = url_for(
            f"{CALLEE_BLP.name}.MicroFrontendStep2", db_id=db_id, _external=True
        )

        # add the next processing step with the data that was stored by the previous step of the invoking plugin
        task: chain = demo_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id="second invoked step",  # name of the next sub-step
            href=href,  # URL to the processing endpoint of the next step
            ui_href=ui_href,  # URL to the micro frontend of the next step
            prog_value=49,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@CALLEE_BLP.route("/<int:db_id>/ui-step-2/")
class MicroFrontendStep2(MethodView):
    """Second micro frontend of the callee plugin."""

    example_inputs = {
        "inputStr": "Callee plugin input string",
    }

    @CALLEE_BLP.html_response(
        HTTPStatus.OK, description="Second micro frontend of the callee plugin."
    )
    @CALLEE_BLP.arguments(
        CalleeInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @CALLEE_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @CALLEE_BLP.html_response(
        HTTPStatus.OK, description="Second Micro frontend of the callee plugin."
    )
    @CALLEE_BLP.arguments(
        CalleeInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @CALLEE_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        plugin = Callee.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = CalleeInputParametersSchema()

        if not data:
            data = {
                "inputStr": "Data from invoking plugin shown in second step: "
                + db_task.data.get("input_str")
            }

        return Response(
            render_template(
                "simple_template.html",
                name=Callee.instance.name,
                version=Callee.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{CALLEE_BLP.name}.ProcessViewStep2", db_id=db_id
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{CALLEE_BLP.name}.MicroFrontendStep2",
                    db_id=db_id,
                    **self.example_inputs,
                ),  # URL of this endpoint
            )
        )


@CALLEE_BLP.route("/<int:db_id>/process-step-2/")
class ProcessViewStep2(MethodView):
    """Start a long running processing task."""

    @CALLEE_BLP.arguments(CalleeInputParametersSchema(unknown=EXCLUDE), location="form")
    @CALLEE_BLP.response(HTTPStatus.OK, CalleeTaskResponseSchema())
    @CALLEE_BLP.require_jwt("jwt", optional=True)
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
        task: chain = demo_task_2.s(db_id=db_task.id) | add_step.s(
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
