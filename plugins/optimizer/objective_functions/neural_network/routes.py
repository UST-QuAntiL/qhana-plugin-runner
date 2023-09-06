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
from typing import Mapping, Optional

from celery.utils.log import get_task_logger
from flask import Response, abort
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
import torch

from plugins.optimizer.interaction_utils.schemas import CallbackUrl, CallbackUrlSchema
from plugins.optimizer.interaction_utils.tasks import make_callback
from plugins.optimizer.objective_functions.neural_network import NN_BLP, NeuralNetwork
from plugins.optimizer.objective_functions.neural_network.neural_network import NN
from plugins.optimizer.objective_functions.neural_network.schemas import (
    HyperparamterInputData,
    HyperparamterInputSchema,
)
from plugins.optimizer.shared.enums import InteractionEndpointType
from plugins.optimizer.shared.schemas import (
    CalcLossOrGradInput,
    CalcLossOrGradInputSchema,
    GradientResponseSchema,
    LossAndGradientResponseData,
    LossAndGradientResponseSchema,
    LossResponseSchema,
    ObjectiveFunctionInvokationCallbackData,
    ObjectiveFunctionInvokationCallbackSchema,
    ObjectiveFunctionPassData,
    ObjectiveFunctionPassDataResponseSchema,
    ObjectiveFunctionPassDataSchema,
    SingleNumpyArraySchema,
)
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InteractionEndpoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from time import perf_counter
from plugins.optimizer.interaction_utils.__init__ import BENCHMARK_LOGGER


TASK_LOGGER = get_task_logger(__name__)


@NN_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @NN_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @NN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=NeuralNetwork.instance.name,
            description=NeuralNetwork.instance.description,
            name=NeuralNetwork.instance.identifier,
            version=NeuralNetwork.instance.version,
            type=PluginType.processing,
            tags=NeuralNetwork.instance.tags,
            entry_point=EntryPoint(
                interaction_endpoints=[
                    InteractionEndpoint(
                        type=InteractionEndpointType.of_pass_data.value,
                        href=url_for(
                            f"{NN_BLP.name}.{PluginsView.__name__}",
                            _external=True,
                        )
                        + "<int:task_id>/pass-data/",
                    ),
                    InteractionEndpoint(
                        type=InteractionEndpointType.calc_loss.value,
                        href=url_for(
                            f"{NN_BLP.name}.{PluginsView.__name__}",
                            _external=True,
                        )
                        + "<int:task_id>/calc-callback-endpoint/",
                    ),
                    InteractionEndpoint(
                        type=InteractionEndpointType.calc_grad.value,
                        href=url_for(
                            f"{NN_BLP.name}.{PluginsView.__name__}",
                            _external=True,
                        )
                        + "<int:task_id>/calc-gradient-endpoint/",
                    ),
                    InteractionEndpoint(
                        type=InteractionEndpointType.calc_grad.value,
                        href=url_for(
                            f"{NN_BLP.name}.{PluginsView.__name__}",
                            _external=True,
                        )
                        + "<int:task_id>/calc-loss-and-grad/",
                    ),
                ],
                href=url_for(f"{NN_BLP.name}.{OptimizerCallbackProcess.__name__}"),
                ui_href=url_for(
                    f"{NN_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
                ),
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


@NN_BLP.route("/ui-hyperparameter/")
class HyperparameterSelectionMicroFrontend(MethodView):
    """Micro frontend for the hyperparameter selection."""

    example_inputs = {
        "number_of_neurons": 10,
    }

    @NN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @NN_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @NN_BLP.arguments(
        CallbackUrlSchema(),
        location="query",
        required=False,
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, callback: CallbackUrl):
        """Return the micro frontend."""
        return self.render(request.args, errors, callback)

    @NN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @NN_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @NN_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, callback: CallbackUrl):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, callback)

    def render(self, data: Mapping, errors: dict, callback: CallbackUrl):
        plugin = NeuralNetwork.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparamterInputSchema()
        callback_schema = CallbackUrlSchema()

        if not data:
            data = {"number_of_neurons": 0.1}
        process_url = url_for(
            f"{NN_BLP.name}.{OptimizerCallbackProcess.__name__}",
            **callback_schema.dump(callback),
        )
        example_values_url = url_for(
            f"{NN_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
            **self.example_inputs,
            **callback_schema.dump(callback),
        )

        help_text = ""

        return Response(
            render_template(
                "simple_template.html",
                name=NeuralNetwork.instance.name,
                version=NeuralNetwork.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text=help_text,
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@NN_BLP.route("/optimizer-callback/")
class OptimizerCallbackProcess(MethodView):
    """Make a callback to the optimizer plugin after selection the hyperparameter."""

    @NN_BLP.arguments(HyperparamterInputSchema(unknown=EXCLUDE), location="form")
    @NN_BLP.arguments(CallbackUrlSchema(unknown=EXCLUDE), location="query", required=True)
    @NN_BLP.response(HTTPStatus.OK, ObjectiveFunctionInvokationCallbackSchema())
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: HyperparamterInputData, callback: CallbackUrl):
        """Start the invoked task."""
        # create new db_task
        db_task = ProcessingTask(
            task_name="neural-network",
        )
        db_task.data["number_of_neurons"] = arguments.number_of_neurons
        db_task.save(commit=True)
        callback_data = ObjectiveFunctionInvokationCallbackSchema().dump(
            ObjectiveFunctionInvokationCallbackData(task_id=db_task.id)
        )

        make_callback(callback.callback_url, callback_data)


@NN_BLP.route("/<int:db_id>/pass-data/")
class PassDataEndpoint(MethodView):
    """Endpoint to add additional data to the db task."""

    @NN_BLP.arguments(
        ObjectiveFunctionPassDataSchema(unknown=EXCLUDE),
        location="json",
        required=True,
    )
    @NN_BLP.response(HTTPStatus.OK, ObjectiveFunctionPassDataResponseSchema())
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: ObjectiveFunctionPassData, db_id: int) -> dict:
        """Endpoint to add additional info to the db task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        serialized_data = ObjectiveFunctionPassDataSchema().dump(input_data)

        db_task.data["x"] = serialized_data["x"]
        db_task.data["y"] = serialized_data["y"]
        number_of_neurons: int = db_task.data["number_of_neurons"]
        number_weights = (
            input_data.x.shape[1] * number_of_neurons
            + number_of_neurons * 1
            + number_of_neurons
            + 1
        )
        db_task.save(commit=True)

        return {"number_weights": number_weights}


@NN_BLP.route("/<int:db_id>/calc-callback-endpoint/")
class CalcCallbackEndpoint(MethodView):
    """Endpoint for the calculation callback."""

    @NN_BLP.response(HTTPStatus.OK, LossResponseSchema())
    @NN_BLP.arguments(
        CalcLossOrGradInputSchema(unknown=EXCLUDE), location="json", required=True
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: CalcLossOrGradInput, db_id: int) -> dict:
        """Endpoint for the calculation callback."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if input_data.x is None:
            input_data.x = (
                SingleNumpyArraySchema().load({"array": db_task.data["x"]}).array
            )
        if input_data.y is None:
            input_data.y = (
                SingleNumpyArraySchema().load({"array": db_task.data["y"]}).array
            )
        number_of_neurons: int = db_task.data["number_of_neurons"]

        nn = NN(input_data.x.shape[1], number_of_neurons)

        nn.set_weights(input_data.x0)

        loss = nn.get_loss(
            torch.tensor(input_data.x, dtype=torch.float32),
            torch.tensor(input_data.y, dtype=torch.float32),
        )
        return {"loss": loss}


@NN_BLP.route("/<int:db_id>/calc-gradient-endpoint/")
class CalcGradientEndpoint(MethodView):
    """Endpoint for the gradient calculation."""

    @NN_BLP.response(HTTPStatus.OK, GradientResponseSchema())
    @NN_BLP.arguments(
        CalcLossOrGradInputSchema(unknown=EXCLUDE), location="json", required=True
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: CalcLossOrGradInput, db_id: int) -> dict:
        """Endpoint for the calculation callback."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if input_data.x is None:
            input_data.x = (
                SingleNumpyArraySchema().load({"array": db_task.data["x"]}).array
            )
        if input_data.y is None:
            input_data.y = (
                SingleNumpyArraySchema().load({"array": db_task.data["y"]}).array
            )

        number_of_neurons: int = db_task.data["number_of_neurons"]

        nn = NN(input_data.x.shape[1], number_of_neurons)

        nn.set_weights(input_data.x0)

        gradient = nn.get_gradient(
            torch.tensor(input_data.x, dtype=torch.float32),
            torch.tensor(input_data.y, dtype=torch.float32),
        )
        return {"gradient": gradient}


@NN_BLP.route("/<int:db_id>/calc-loss-and-grad/")
class CalcLossandGradEndpoint(MethodView):
    """Endpoint for the gradient calculation."""

    @NN_BLP.response(HTTPStatus.OK, LossAndGradientResponseSchema())
    @NN_BLP.arguments(
        CalcLossOrGradInputSchema(unknown=EXCLUDE), location="json", required=True
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: CalcLossOrGradInput, db_id: int) -> dict:
        """Endpoint for the calculation callback."""

        bench_start_ofcalc = perf_counter()
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if input_data.x is None:
            input_data.x = (
                SingleNumpyArraySchema().load({"array": db_task.data["x"]}).array
            )
        if input_data.y is None:
            input_data.y = (
                SingleNumpyArraySchema().load({"array": db_task.data["y"]}).array
            )

        number_of_neurons: int = db_task.data["number_of_neurons"]

        nn = NN(input_data.x.shape[1], number_of_neurons)

        nn.set_weights(input_data.x0)

        loss, grad = nn.get_loss_and_gradient(
            torch.tensor(input_data.x, dtype=torch.float32),
            torch.tensor(input_data.y, dtype=torch.float32),
        )

        response = LossAndGradientResponseSchema().dump(
            LossAndGradientResponseData(loss=loss, gradient=grad)
        )
        bench_stop_ofcalc = perf_counter()
        bench_diff_ofcalc = bench_stop_ofcalc - bench_start_ofcalc
        BENCHMARK_LOGGER.info(f"bench_diff_of_calcnet: {bench_diff_ofcalc}")

        return response
