# Copyright 2023 QHAna plugin runner contributors.
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

import numpy as np
import requests
from celery import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from plugins.optimizer.coordinator.tasks import echo_results, of_pass_data
from plugins.optimizer.interaction_utils.tasks import invoke_task
from plugins.optimizer.shared.schemas import (
    MinimizerCallbackData,
    MinimizerCallbackSchema,
    MinimizerInputData,
    MinimizerInputSchema,
    ObjectiveFunctionInvokationCallbackData,
    ObjectiveFunctionInvokationCallbackSchema,
    TaskStatusChanged,
    TaskStatusChangedSchema,
)
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InteractionEndpointType,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import OPTIMIZER_BLP, Optimizer
from .schemas import OptimizerSetupTaskInputData, OptimizerSetupTaskInputSchema


def get_plugin_metadata(plugin_url) -> PluginMetadata:
    """
    Get the metadata of a plugin.

    Args:
        plugin_url (str): The URL of the plugin.

    Returns:
        PluginMetadata: The metadata of the plugin.
    """

    plugin_metadata = requests.get(plugin_url).json()
    schema = PluginMetadataSchema()
    metadata: PluginMetadata = schema.load(plugin_metadata)
    return metadata


@OPTIMIZER_BLP.route("/")
class MetadataView(MethodView):
    """
    Plugins collection resource.

    A View that handles the plugins metadata.
    """

    @OPTIMIZER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self) -> PluginMetadata:
        """
        Optimizer endpoint returning the plugin metadata.

        Returns:
            PluginMetadata: The metadata of the optimizer plugin.
        """
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
    """
    Micro frontend for selection of:
    1. objective-function plugin
    2. dataset
    3. minimizer plugin
    4. target variable

    This class is responsible for the handling of the setup UI.
    """

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
    def get(self, errors) -> Response:
        """
        Return the micro frontend.

        Args:
            errors (dict): A dictionary containing possible errors.

        Returns:
            A rendered template of the micro frontend.
        """

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
    def post(self, errors) -> Response:
        """
        Return the micro frontend with prerendered inputs.

        Args:
            errors (dict): A dictionary containing possible errors.

        Returns:
            A rendered template of the micro frontend with prerendered inputs.
        """

        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict) -> Response:
        """
        Render the UI for the plugin setup.

        Args:
            data (Mapping): The input data for the setup.
            errors (dict): A dictionary containing possible errors.

        Returns:
            A rendered template of the plugin setup UI.
        """
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
    """
    UI input processing ressource.

    A View that handles handles the processing of the UI input data.
    """

    @OPTIMIZER_BLP.arguments(
        OptimizerSetupTaskInputSchema(unknown=EXCLUDE), location="form"
    )
    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: OptimizerSetupTaskInputData) -> Response:
        """Handle POST requests for the setup UI.

        This method handles the POST requests for the setup UI.
        It saves the input data to the database, gets the metadata of the plugins and saves the endpoints to the database.
        It also invokes the objective function plugin.

        Args:
            arguments (OptimizerSetupTaskInputData): The input data of the setup UI.

        Returns:
            Response: A redirect to the task view.
        """
        db_task = ProcessingTask(
            task_name="optimizer_setup",
        )

        # save the input data to the database
        db_task.data["input_file_url"] = arguments.input_file_url
        db_task.data["target_variable"] = arguments.target_variable
        db_task.clear_previous_step()
        db_task.save(commit=True)

        # save the callback endpoint for the minimizer plugins to the database
        min_callback_url = url_for(
            f"{OPTIMIZER_BLP.name}.{MinimizerSetupCallback.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["minimize_callback_url"] = min_callback_url

        # get the minimizer plugin metadata and save the endpoints to the database
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

        # get the objective function plugin metadata and save the endpoints to the database
        # also invoke the plugin
        of_callback_url = url_for(
            f"{OPTIMIZER_BLP.name}.{ObjectiveFunctionInvokationCallback.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        of_plugin_metadata: PluginMetadata = get_plugin_metadata(
            arguments.objective_function_plugin_selector
        )

        of_calc_endpoint = [
            element
            for element in of_plugin_metadata.entry_point.interaction_endpoints
            if element.type == InteractionEndpointType.objective_function_calc
        ]

        db_task.data[
            InteractionEndpointType.objective_function_calc.value
        ] = of_calc_endpoint[0].href

        of_pass_data_endpoint = [
            element
            for element in of_plugin_metadata.entry_point.interaction_endpoints
            if element.type == InteractionEndpointType.of_pass_data
        ]

        db_task.data[InteractionEndpointType.of_pass_data.value] = of_pass_data_endpoint[
            0
        ].href

        of_gradient_endpoint = [
            element
            for element in of_plugin_metadata.entry_point.interaction_endpoints
            if element.type == InteractionEndpointType.objective_function_gradient
        ]

        if len(of_gradient_endpoint) > 0:
            db_task.data[
                InteractionEndpointType.objective_function_gradient.value
            ] = of_gradient_endpoint[0].href

        of_href = urljoin(
            arguments.objective_function_plugin_selector,
            of_plugin_metadata.entry_point.href,
        )

        of_ui_href = urljoin(
            arguments.objective_function_plugin_selector,
            of_plugin_metadata.entry_point.ui_href,
        )

        db_task.save(commit=True)

        task = invoke_task.s(
            db_id=db_task.id,
            step_id="objective function plugin setup",
            href=of_href,
            ui_href=of_ui_href,
            callback_url=of_callback_url,
            prog_value=25,
            task_log="hyperparameter selection started",
        )

        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTIMIZER_BLP.route("/<int:db_id>/objective-function-invokation-callback/")
class ObjectiveFunctionInvokationCallback(MethodView):
    """
    Callback for the objective function.

    This class handles the callbacks for the objective function invokation.
    """

    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.arguments(
        ObjectiveFunctionInvokationCallbackSchema(unknown=EXCLUDE), location="json"
    )
    def post(self, arguments: ObjectiveFunctionInvokationCallbackData, db_id: int):
        """
        Handle POST requests for the objective function callback.

        This method handles the POST requests for the objective function callback.
        It saves the objective function result to the database,
        passes x and y to the objective function plugin and invokes the minimizer plugin.

        Args:
            callback_data (ObjectiveFunctionInvokationCallbackData): The callback data from the objective function invokation.
            task_id (str): The ID of the task.

        Returns:
            TaskStatusChanged: The changed status of the task.
        """
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.data["of_db_id"] = arguments.db_id

        db_task.clear_previous_step()
        db_task.save(commit=True)

        min_href = db_task.data["min_href"]
        min_ui_href = db_task.data["min_ui_href"]
        min_callback_url = db_task.data["minimize_callback_url"]

        task: chain = of_pass_data.s(db_id=db_task.id) | invoke_task.s(
            db_id=db_task.id,
            step_id="minimization plugin setup",
            href=min_href,
            ui_href=min_ui_href,
            callback_url=min_callback_url,
            prog_value=50,
        )

        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTIMIZER_BLP.route("/<int:db_id>/minimizer-setup-callback/")
class MinimizerSetupCallback(MethodView):
    """
    Callback for the minimizer.

    This class handles the callbacks for the minimizer.
    """

    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.arguments(MinimizerCallbackSchema(unknown=EXCLUDE), location="json")
    def post(self, arguments: MinimizerCallbackData, db_id: int):
        """
        Handle POST requests for the minimizer callback.

        This method handles the POST requests for the minimizer callback.
        It saves the minimizer result to the database and invokes the minization process of the minimizer plugin.

        Args:
            callback_data (MinimizerCallbackData): The callback data from the minimizer.
            task_id (str): The ID of the task.
        """
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.progress_value = 75
        db_task.clear_previous_step()
        db_task.save(commit=True)

        calc_loss_endpoint_url: str = db_task.data.get(
            InteractionEndpointType.objective_function_calc.value
        )

        calc_gradient_endpoint_url: str = db_task.data.get(
            InteractionEndpointType.objective_function_gradient.value
        )

        of_db_id: str = db_task.data.get("of_db_id")

        calc_loss_endpoint_url = calc_loss_endpoint_url.replace(
            "<int:db_id>", str(of_db_id)
        )

        if calc_gradient_endpoint_url:
            calc_gradient_endpoint_url = calc_gradient_endpoint_url.replace(
                "<int:db_id>", str(of_db_id)
            )
        number_weights = db_task.data.get("number_weights")
        x0 = np.random.randn(number_weights)

        min_input_data = MinimizerInputSchema().dump(
            MinimizerInputData(
                calc_loss_endpoint_url=calc_loss_endpoint_url,
                calc_gradient_endpoint_url=calc_gradient_endpoint_url,
                x0=x0,
                callback_url=url_for(
                    f"{OPTIMIZER_BLP.name}.{MinimizerResultCallback.__name__}",
                    db_id=db_task.id,
                    _external=True,
                ),
            )
        )

        response = requests.post(arguments.minimize_endpoint_url, json=min_input_data)

        response.raise_for_status()


@OPTIMIZER_BLP.route("/<int:db_id>/minimizer-result-callback/")
class MinimizerResultCallback(MethodView):
    """Callback endpoint for the minimizer plugin after minimization."""

    @OPTIMIZER_BLP.response(HTTPStatus.OK)
    @OPTIMIZER_BLP.arguments(TaskStatusChangedSchema(unknown=EXCLUDE), location="json")
    def post(self, arguments: TaskStatusChanged, db_id: int) -> Response:
        """
        Handle POST requests for the minimizer result callback.

        This method accepts callback data from the minimizer's operation and saves the result to a file.

        Args:
            callback_data (MinimizerResultCallbackData): The callback data from the minimizer result.
            task_id (str): The ID of the task.

        Returns:
            A redirect to the task view.
        """
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if arguments.status != "SUCCESS" or arguments.url is None:
            db_task.status = "FAILURE"
            db_task.save(commit=True)
            return

        db_task.data["minimizer_result_url"] = arguments.url
        db_task.clear_previous_step()
        db_task.save(commit=True)

        task: chain = echo_results.s(db_id=db_id) | save_task_result.s(db_id=db_id)
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()
        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
