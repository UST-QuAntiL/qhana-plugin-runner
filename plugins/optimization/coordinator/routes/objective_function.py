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
    ObjectiveFunctionCallbackSchema,
    ObjectiveFunctionCallbackData,
    CallbackURL,
    CallbackURLSchema,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error
from .dataset_and_start import StartOptimization, DatasetSelectionUI
from .. import OPTI_COORD_BLP, OptimizationCoordinator
from ..schemas import (
    ObjFuncSelectionSchema,
    TaskResponseSchema,
    InternalDataSchema,
    InternalData,
    ObjFuncSelectionData,
)
from ..tasks import no_op_task


@OPTI_COORD_BLP.route("/<int:db_id>/obj-func-selection/")
class ObjFuncSelectionUI(MethodView):
    """Micro frontend for the selection of the objective function plugin."""

    # FIXME: remove when plugin selection PR has been merged
    example_inputs = {
        "objectiveFunctionUrl": "http://localhost:5005/plugins/objective-function-demo%40v0-1-0",
    }

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the objective function plugin.",
    )
    @OPTI_COORD_BLP.arguments(
        ObjFuncSelectionSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id)

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the objective function plugin.",
    )
    @OPTI_COORD_BLP.arguments(
        ObjFuncSelectionSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id)

    def render(self, data: Mapping, errors: dict, db_id: int):
        schema = ObjFuncSelectionSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=OptimizationCoordinator.instance.name,
                version=OptimizationCoordinator.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTI_COORD_BLP.name}.{ObjFuncSetupProcess.__name__}", db_id=db_id
                ),  # URL of the first processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OPTI_COORD_BLP.name}.{ObjFuncSelectionUI.__name__}",  # URL of this endpoint,
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@OPTI_COORD_BLP.route("/<int:db_id>/obj-func-setup/")
class ObjFuncSetupProcess(MethodView):
    """
    Adds the setup of the objective function plugin as the next step. This leads to the micro frontend of the objective
    function plugin to be displayed to the user who can then input the required hyperparameters.
    """

    @OPTI_COORD_BLP.arguments(ObjFuncSelectionSchema(unknown=EXCLUDE), location="form")
    @OPTI_COORD_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: ObjFuncSelectionData, db_id: int):
        """Start the objective function setup task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()

        schema = InternalDataSchema()
        internal_data: InternalData = schema.loads(db_task.parameters)

        internal_data.objective_function_url = arguments.objective_function_url

        db_task.parameters = schema.dumps(internal_data)
        db_task.save(commit=True)

        # add new step where the setup of the objective function plugin is executed

        # name of the next step
        step_id = "objective-function-setup"

        # get metadata
        schema = PluginMetadataSchema()
        resp = requests.get(internal_data.objective_function_url)

        if resp.status_code >= 400:
            TASK_LOGGER.error(f"{resp.status_code} {resp.reason} {resp.text}")

        raw_metadata = resp.json()
        plugin_metadata: PluginMetadata = schema.load(raw_metadata)

        # URL of the process endpoint of the objective function plugin
        href = urljoin(
            internal_data.objective_function_url, plugin_metadata.entry_point.href
        )
        # URL of the micro frontend endpoint of the objective function plugin
        ui_href = urljoin(
            internal_data.objective_function_url, plugin_metadata.entry_point.ui_href
        )

        callback_schema = CallbackURLSchema()
        callback_url = CallbackURL(
            callback_url=url_for(
                f"{OPTI_COORD_BLP.name}.{ObjFuncCallback.__name__}",
                db_id=str(db_task.id),
                _external=True,
            )
        )

        callback_url_query = urlencode(callback_schema.dump(callback_url))

        # Chain a processing task (that does nothing) with executing the objective function plugin setup.
        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            # adds callback URL so that the objective function plugin can signal that it has finished and send data back
            ui_href=ui_href + "?" + callback_url_query,
            prog_value=60,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTI_COORD_BLP.route("/<int:db_id>/obj-func-callback/")
class ObjFuncCallback(MethodView):
    """
    Callback endpoint that will be used by the objective function plugins to signal that they have finished and to send
    data back.
    """

    @OPTI_COORD_BLP.arguments(ObjectiveFunctionCallbackSchema(unknown=EXCLUDE))
    @OPTI_COORD_BLP.response(HTTPStatus.OK)
    def post(self, arguments: ObjectiveFunctionCallbackData, db_id: int):
        """
        Gets data from the objective function plugin and starts the next step.

        :param arguments:
        :param db_id: Database ID of the task object. This refers to the database that the objective function plugin uses which might not be the same as the one this plugin uses.
        :return:
        """
        TASK_LOGGER.info("Objective function setup callback called")

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()

        schema = InternalDataSchema()
        internal_data: InternalData = schema.loads(db_task.parameters)

        internal_data.objective_function_calculation_url = arguments.calculation_endpoint
        internal_data.number_of_parameters = arguments.number_of_parameters

        db_task.parameters = schema.dumps(internal_data)
        db_task.save(commit=True)

        step_id = "dataset-selection"
        href = url_for(
            f"{OPTI_COORD_BLP.name}.{StartOptimization.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{OPTI_COORD_BLP.name}.{DatasetSelectionUI.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=80,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()
