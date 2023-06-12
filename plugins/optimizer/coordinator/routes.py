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
from urllib.parse import urljoin

import requests
from celery import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from plugins.optimizer.coordinator.tasks import optimize_task
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
from .schemas import OptimizerSetupTaskInputData, OptimizerSetupTaskInputSchema
from .shared_schemas import (
    MinimizerCallbackData,
    MinimizerCallbackSchema,
    ObjectiveFunctionCallbackData,
    ObjectiveFunctionCallbackSchema,
)


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
                ),
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
    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: OptimizerSetupTaskInputData):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name="optimizer_setup",
        )

        db_task.data["input_file_url"] = arguments.input_file_url
        db_task.data["target_variable"] = arguments.target_variable
        db_task.clear_previous_step()
        db_task.save(commit=True)

        min_callback_url = url_for(
            f"{OPTIMIZER_BLP.name}.{MinimizeCallback.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["minimize_callback_url"] = min_callback_url

        min_plugin_metadata: PluginMetadata = get_plugin_metadata(
            arguments.minimizer_plugin_selector
        )

        min_href = urljoin(
            arguments.minimizer_plugin_selector, min_plugin_metadata.entry_point.href
        )

        db_task.data["min_href"] = min_href

        min_ui_href = urljoin(
            arguments.minimizer_plugin_selector,
            min_plugin_metadata.entry_point.ui_href,
        )

        db_task.data["min_ui_href"] = min_ui_href

        db_task.save(commit=True)

        of_callback_url = url_for(
            f"{OPTIMIZER_BLP.name}.{ObjectiveFunctionCallback.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        # invoke the selected objective function plugin for hyperparameter selection

        of_plugin_metadata: PluginMetadata = get_plugin_metadata(
            arguments.objective_function_plugin_selector
        )

        of_href = urljoin(
            arguments.objective_function_plugin_selector,
            of_plugin_metadata.entry_point.href,
        )

        of_ui_href = urljoin(
            arguments.objective_function_plugin_selector,
            of_plugin_metadata.entry_point.ui_href,
        )

        task = invoke_task.s(
            db_id=db_task.id,
            step_id="of_setup",
            href=of_href,
            ui_href=of_ui_href,
            callback_url=of_callback_url,
            prog_value=20,
            task_log="hyperparameter selection started",
        )

        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTIMIZER_BLP.route("/<int:db_id>/objective-function-callback/")
class ObjectiveFunctionCallback(MethodView):
    """Callback function for the objective-function plugin."""

    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.arguments(
        ObjectiveFunctionCallbackSchema(unknown=EXCLUDE), location="json"
    )
    def post(self, arguments: ObjectiveFunctionCallbackData, db_id: int):
        """starts the next step of the optimizer plugin"""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if arguments.hyperparameters is not None:
            db_task.data["hyperparameters"] = arguments.hyperparameters

        db_task.data["calc_loss_endpoint_url"] = arguments.calc_loss_endpoint_url

        db_task.clear_previous_step()
        db_task.save(commit=True)

        min_href = db_task.data["min_href"]
        min_ui_href = db_task.data["min_ui_href"]
        min_callback_url = db_task.data["minimize_callback_url"]

        task = invoke_task.s(
            db_id=db_task.id,
            step_id="min_setup",
            href=min_href,
            ui_href=min_ui_href,
            callback_url=min_callback_url,
            prog_value=40,
            task_log="minimizer setup started",
        )

        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTIMIZER_BLP.route("/<int:db_id>/minimizer-callback/")
class MinimizeCallback(MethodView):
    """Callback function for the minimizer plugin."""

    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.arguments(MinimizerCallbackSchema(unknown=EXCLUDE), location="json")
    def post(self, arguments: MinimizerCallbackData, db_id: int):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.data["minimize_endpoint_url"] = arguments.minimize_endpoint_url
        db_task.progress_value = 60
        db_task.clear_previous_step()
        db_task.save(commit=True)

        task: chain = optimize_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )

        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
