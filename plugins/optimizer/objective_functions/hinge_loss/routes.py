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
from typing import Mapping

from celery.utils.log import get_task_logger
from flask import Response, abort, render_template, request, url_for
from flask.views import MethodView
from marshmallow import EXCLUDE

from plugins.optimizer.interaction_utils.schemas import CallbackUrl, CallbackUrlSchema
from plugins.optimizer.interaction_utils.tasks import make_callback
from plugins.optimizer.shared.enums import InteractionEndpointType
from plugins.optimizer.shared.schemas import (
    CalcLossOrGradInput,
    CalcLossOrGradInputSchema,
    LossResponseSchema,
    ObjectiveFunctionInvokationCallbackData,
    ObjectiveFunctionInvokationCallbackSchema,
    ObjectiveFunctionPassData,
    ObjectiveFunctionPassDataResponseSchema,
    ObjectiveFunctionPassDataSchema,
)
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InteractionEndpoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)

from . import HINGELOSS_BLP, HingeLoss
from .schemas import HyperparamterInputData, HyperparamterInputSchema
from .tasks import hinge_loss

TASK_LOGGER = get_task_logger(__name__)


@HINGELOSS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @HINGELOSS_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=HingeLoss.instance.name,
            description=HingeLoss.instance.description,
            name=HingeLoss.instance.identifier,
            version=HingeLoss.instance.version,
            type=PluginType.processing,
            tags=HingeLoss.instance.tags,
            entry_point=EntryPoint(
                interaction_endpoints=[
                    InteractionEndpoint(
                        type=InteractionEndpointType.of_pass_data.value,
                        href=url_for(
                            f"{HINGELOSS_BLP.name}.{PassDataEndpoint.__name__}",
                            _external=True,
                        ),
                    ),
                    InteractionEndpoint(
                        type=InteractionEndpointType.calc_loss.value,
                        href=url_for(
                            f"{HINGELOSS_BLP.name}.{CalcCallbackEndpoint.__name__}",
                            _external=True,
                        ),
                    ),
                ],
                href=url_for(f"{HINGELOSS_BLP.name}.{OptimizerCallbackProcess.__name__}"),
                ui_href=url_for(
                    f"{HINGELOSS_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
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


@HINGELOSS_BLP.route("/ui/")
class HyperparameterSelectionMicroFrontend(MethodView):
    """Micro frontend for the hyperparameter selection."""

    example_inputs = {
        "c": 1.0,
    }

    @HINGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @HINGELOSS_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @HINGELOSS_BLP.arguments(
        CallbackUrlSchema(),
        location="query",
        required=False,
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, callback: CallbackUrl):
        """Return the micro frontend."""
        return self.render(request.args, errors, callback)

    @HINGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @HINGELOSS_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @HINGELOSS_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, callback: CallbackUrl):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, callback)

    def render(self, data: Mapping, errors: dict, callback: CallbackUrl):
        plugin = HingeLoss.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparamterInputSchema()
        callback_schema = CallbackUrlSchema()

        if not data:
            data = {"c": 1.0}
        process_url = url_for(
            f"{HINGELOSS_BLP.name}.{OptimizerCallbackProcess.__name__}",
            **callback_schema.dump(callback),
        )
        example_values_url = url_for(
            f"{HINGELOSS_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
            **self.example_inputs,
            **callback_schema.dump(callback),
        )

        return Response(
            render_template(
                "simple_template.html",
                name=HingeLoss.instance.name,
                version=HingeLoss.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text="The Regularization Strength of the hinge loss.",
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@HINGELOSS_BLP.route("/process/")
class OptimizerCallbackProcess(MethodView):
    """Save the hyperparameters to the database and make a callback to the optimizer."""

    @HINGELOSS_BLP.arguments(HyperparamterInputSchema(unknown=EXCLUDE), location="form")
    @HINGELOSS_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE), location="query", required=True
    )
    @HINGELOSS_BLP.response(HTTPStatus.OK, ObjectiveFunctionInvokationCallbackSchema())
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: HyperparamterInputData, callback: CallbackUrl):
        """Start the invoked task."""

        hyperparameters = {
            "c": arguments.c,
        }
        callback_data = ObjectiveFunctionInvokationCallbackSchema().dump(
            ObjectiveFunctionInvokationCallbackData(hyperparameters=hyperparameters)
        )

        make_callback(callback.callback_url, callback_data)


@HINGELOSS_BLP.route("/pass-data/")
class PassDataEndpoint(MethodView):
    """Endpoint to add additional data to the db task."""

    @HINGELOSS_BLP.arguments(
        ObjectiveFunctionPassDataSchema(unknown=EXCLUDE),
        location="json",
        required=True,
    )
    @HINGELOSS_BLP.response(HTTPStatus.OK, ObjectiveFunctionPassDataResponseSchema())
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: ObjectiveFunctionPassData) -> dict:
        """Endpoint to add additional info to the db task."""

        return {"number_weights": input_data.x.shape[1]}


@HINGELOSS_BLP.route("/calc-loss/")
class CalcCallbackEndpoint(MethodView):
    """Endpoint for the calculation callback."""

    @HINGELOSS_BLP.response(HTTPStatus.OK, LossResponseSchema())
    @HINGELOSS_BLP.arguments(
        CalcLossOrGradInputSchema(unknown=EXCLUDE), location="json", required=True
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: CalcLossOrGradInput) -> dict:
        """Endpoint for the calculation callback."""

        loss = hinge_loss(
            X=input_data.x,
            y=input_data.y,
            w=input_data.x0,
            C=input_data.hyperparameters["c"],
        )
        return {"loss": loss}
