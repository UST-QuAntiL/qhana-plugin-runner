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
from logging import Logger
from typing import Mapping, Optional
from urllib.parse import urljoin, urlencode

import requests
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginMetadataSchema,
    OptimizerCallbackSchema,
    OptimizerCallbackData,
    CallbackURLSchema,
    CallbackURL,
)
from qhana_plugin_runner.api.tasks_api import TaskStatusSchema, TaskStatus
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error
from .objective_function import ObjFuncSetupProcess, ObjFuncSelectionUI
from .. import OPTI_COORD_BLP, OptimizationCoordinator
from ..schemas import (
    ObjFuncSelectionSchema,
    TaskResponseSchema,
    OptimSelectionSchema,
    InternalDataSchema,
    InternalData,
    OptimSelectionData,
)
from ..tasks import no_op_task


@OPTI_COORD_BLP.route("/optim-selection/")
class OptimSelectionUI(MethodView):
    """Micro frontend for the selection of the optimizer plugin."""

    # FIXME: remove when plugin selection PR has been merged
    example_inputs = {
        "optimizerUrl": "http://localhost:5005/plugins/optimizer-demo%40v0-1-0",
    }

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the optimizer plugin.",
    )
    @OPTI_COORD_BLP.arguments(
        ObjFuncSelectionSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the optimizer plugin.",
    )
    @OPTI_COORD_BLP.arguments(
        ObjFuncSelectionSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = OptimSelectionSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=OptimizationCoordinator.instance.name,
                version=OptimizationCoordinator.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTI_COORD_BLP.name}.{OptimSetupProcess.__name__}"
                ),  # URL of the first processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OPTI_COORD_BLP.name}.{OptimSelectionUI.__name__}",  # URL of this endpoint
                    **self.example_inputs,
                ),
            )
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@OPTI_COORD_BLP.route("/optim-setup/")
class OptimSetupProcess(MethodView):
    """
    Adds the setup of the optimizer plugin as the next step. This leads to the micro frontend of the optimizer plugin
    to be displayed to the user who can then input the required hyperparameters.
    """

    @OPTI_COORD_BLP.arguments(OptimSelectionSchema(unknown=EXCLUDE), location="form")
    @OPTI_COORD_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: OptimSelectionData):
        """Start the optimizer setup task."""
        db_task = ProcessingTask(
            task_name=no_op_task.name,
        )
        db_task.save(commit=True)

        schema = InternalDataSchema()
        internal_data: InternalData = InternalData()

        internal_data.optimizer_plugin_url = arguments.optimizer_url

        db_task.parameters = schema.dumps(internal_data)
        db_task.save(commit=True)

        # add new step where the setup of the optimizer plugin is executed

        # name of the next step
        step_id = "optimizer-setup"

        # get metadata
        metadata_schema = PluginMetadataSchema()
        resp = requests.get(internal_data.optimizer_plugin_url)

        if resp.status_code >= 400:
            TASK_LOGGER.error(
                f"{resp.request.url} {resp.status_code} {resp.reason} {resp.text}"
            )

        raw_metadata = resp.json()
        plugin_metadata: PluginMetadata = metadata_schema.load(raw_metadata)

        # URL of the process endpoint of the objective function plugin
        href = urljoin(
            internal_data.optimizer_plugin_url, plugin_metadata.entry_point.href
        )
        # URL of the micro frontend endpoint of the objective function plugin
        ui_href = urljoin(
            internal_data.optimizer_plugin_url, plugin_metadata.entry_point.ui_href
        )

        callback_schema = CallbackURLSchema()
        callback_url = CallbackURL(
            callback_url=url_for(
                f"{OPTI_COORD_BLP.name}.{OptimCallback.__name__}",
                db_id=str(db_task.id),
                _external=True,
            )
        )

        callback_url_query = urlencode(callback_schema.dump(callback_url))

        # Chain a processing task (that does nothing) with executing the optimization plugin setup.
        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            # adds callback URL so that the optimization plugin can signal that it has finished and send data back
            ui_href=ui_href + "?" + callback_url_query,
            prog_value=20,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTI_COORD_BLP.route("/<int:db_id>/optim-callback/")
class OptimCallback(MethodView):
    """
    Callback endpoint that will be used by the optimizer plugins to signal that they have finished and to send
    data back.
    """

    @OPTI_COORD_BLP.arguments(OptimizerCallbackSchema(unknown=EXCLUDE))
    @OPTI_COORD_BLP.response(HTTPStatus.OK)
    def post(self, arguments: OptimizerCallbackData, db_id: int):
        """
        Gets data from the objective function plugin and starts the next step.

        :param arguments:
        :param db_id: Database ID of the task object. This refers to the database that the objective function plugin uses which might not be the same as the one this plugin uses.
        :return:
        """
        TASK_LOGGER.info("Optimizer setup callback called")

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()

        task_schema = TaskStatusSchema()
        task_status: TaskStatus = task_schema.load(
            requests.get(arguments.task_url).json()
        )

        schema = InternalDataSchema()
        internal_data: InternalData = schema.loads(db_task.parameters)

        optimizer_start_url = task_status.steps[-1].href

        internal_data.optimizer_start_url = optimizer_start_url

        db_task.parameters = schema.dumps(internal_data)
        db_task.save(commit=True)

        step_id = "obj-func-selection"
        href = url_for(
            f"{OPTI_COORD_BLP.name}.{ObjFuncSetupProcess.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{OPTI_COORD_BLP.name}.{ObjFuncSelectionUI.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=40,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()
