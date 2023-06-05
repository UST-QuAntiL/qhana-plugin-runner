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
from urllib.parse import unquote, urljoin

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
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import invoke_task, save_task_error, save_task_result

from . import OPTIMIZER_BLP, Optimizer
from .schemas import (
    ObjectiveFunctionHyperparameterCallbackData,
    ObjectiveFunctionHyperparameterCallbackSchema,
    OptimizerSetupTaskInputSchema,
    OptimizerTaskResponseSchema,
)
from .tasks import minimize_task


def get_plugin_metadata(plugin_url) -> PluginMetadata:
    """Get the metadata of a plugin.

    Args:
        plugin_name (str): The name of the plugin.

    Returns:
        PluginMetadata: The metadata of the plugin.
    """
    plugin_metadata = requests.get(plugin_url).json()
    schema = PluginMetadataSchema()
    metadata: PluginMetadata = schema.load(plugin_metadata)
    return metadata


@OPTIMIZER_BLP.route("/")
class MetadataView(MethodView):
    """Plugins collection resource."""

    @OPTIMIZER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Optimizer endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Optimizer plugin",
            description=Optimizer.instance.description,
            name=Optimizer.instance.name,
            version=Optimizer.instance.version,
            type=PluginType.processing,
            tags=Optimizer.instance.tags,
            entry_point=EntryPoint(
                href=url_for(
                    f"{OPTIMIZER_BLP.name}.{OptimizerSetupProcessStep.__name__}"
                ),  # URL for the first process endpoint
                ui_href=url_for(
                    f"{OPTIMIZER_BLP.name}.{OptimizerSetupMicroFrontend.__name__}"
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


@OPTIMIZER_BLP.route("/ui-setup/")
class OptimizerSetupMicroFrontend(MethodView):
    """Micro frontend for the objective-function and dataset selection in the optimizer plugin."""

    example_inputs = {}

    @OPTIMIZER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the objective-function and dataset selection in the optimizer plugin.",
    )
    @OPTIMIZER_BLP.arguments(
        OptimizerSetupTaskInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @OPTIMIZER_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the objective-function and dataset selection in the optimizer plugin.",
    )
    @OPTIMIZER_BLP.arguments(
        OptimizerSetupTaskInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = OptimizerSetupTaskInputSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=Optimizer.instance.name,
                version=Optimizer.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTIMIZER_BLP.name}.{OptimizerSetupProcessStep.__name__}"
                ),  # URL of the first processing step
                example_values=url_for(
                    f"{OPTIMIZER_BLP.name}.{OptimizerSetupMicroFrontend.__name__}",
                    **self.example_inputs,
                ),
            )
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@OPTIMIZER_BLP.route("/process-setup/")
class OptimizerSetupProcessStep(MethodView):
    """Start the process step of the objective-function selection."""

    @OPTIMIZER_BLP.arguments(
        OptimizerSetupTaskInputSchema(unknown=EXCLUDE), location="form"
    )
    @OPTIMIZER_BLP.response(HTTPStatus.OK, OptimizerTaskResponseSchema())
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name="hyperparameter_selection",
        )
        db_task.save(commit=True)

        # extract the plugin name from the plugin url
        plugin_url = arguments["objective_function_plugin_selector"]

        # save the input file url to the database
        db_task.data["input_file_url"] = arguments["input_file_url"]

        db_task.data["target_variable"] = arguments["target_variable"]

        db_task.save(commit=True)

        # add new step where the objective-function plugin is executed

        # name of the next step
        step_id = "hyperparamter selection"
        # URL of the process endpoint of the invoked plugin
        callback_url = url_for(
            f"{OPTIMIZER_BLP.name}.{ObjectiveFunctionCallback.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        callback_url = unquote(callback_url)

        plugin_metadata: PluginMetadata = get_plugin_metadata(plugin_url)

        href = urljoin(plugin_url, plugin_metadata.entry_point.href)
        # URL of the micro frontend endpoint of the invoked plugin
        ui_href = urljoin(plugin_url, plugin_metadata.entry_point.ui_href)

        # Chain the first processing task with executing the objective-function plugin.
        # All tasks use the same db_id to be able to fetch data from the previous steps and to store data for the next
        # steps in the database.
        task = invoke_task.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            callback_url=callback_url,
            prog_value=20,
            task_log="hyperparameter selection step added",
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTIMIZER_BLP.route("/<int:db_id>/objective-function-callback/")
class ObjectiveFunctionCallback(MethodView):
    """Callback function for the objective-function plugin."""

    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.arguments(
        ObjectiveFunctionHyperparameterCallbackSchema(unknown=EXCLUDE), location="json"
    )
    def post(self, arguments: ObjectiveFunctionHyperparameterCallbackData, db_id: int):
        """starts the next step of the optimizer plugin"""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if arguments.hyperparameters != None:
            db_task.data["hyperparameters"] = arguments.hyperparameters

        calc_url = arguments.calc_loss_enpoint_url
        db_task.data["calc_loss_endpoint_url"] = calc_url

        db_task.clear_previous_step()
        db_task.save(commit=True)

        # add new step where the objective-function plugin is executed

        # name of the next step
        step_id = "minimization step"
        # URL of the process endpoint of the invoked plugin

        task: chain = minimize_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_id)

        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()
