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
from flask import Response, abort, render_template, request, url_for
from flask.views import MethodView
from marshmallow import EXCLUDE
from .interaction_utils.ie_utils import url_for_ie
from .interaction_utils.schemas import CallbackUrl, CallbackUrlSchema
from .interaction_utils.tasks import make_callback
from .shared.enums import InteractionEndpointType
from .shared.schemas import (
    CalcLossOrGradInput,
    CalcLossOrGradInputSchema,
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
                        # Since the endpoint has the task id in the url, we need to add a placeholder
                        href=url_for_ie(
                            f"{HINGELOSS_BLP.name}.{PassDataEndpoint.__name__}"
                        ),
                    ),
                    InteractionEndpoint(
                        type=InteractionEndpointType.calc_loss.value,
                        href=url_for_ie(
                            f"{HINGELOSS_BLP.name}.{CalcLossEndpoint.__name__}"
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


@HINGELOSS_BLP.route("/ui-hyperparameter/")
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

        # set default values if not present
        if not data:
            data = {"c": 1.0}
        process_url = url_for(
            f"{HINGELOSS_BLP.name}.{OptimizerCallbackProcess.__name__}",
            # forward the callback url to the processing step
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


@HINGELOSS_BLP.route("/optimizer-callback/")
class OptimizerCallbackProcess(MethodView):
    """Save the hyperparameters to the database and make a callback to the coordinator."""

    @HINGELOSS_BLP.arguments(HyperparamterInputSchema(unknown=EXCLUDE), location="form")
    @HINGELOSS_BLP.arguments(
        CallbackUrlSchema(unknown=EXCLUDE), location="query", required=True
    )
    @HINGELOSS_BLP.response(HTTPStatus.OK, ObjectiveFunctionInvokationCallbackSchema())
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: HyperparamterInputData, callback: CallbackUrl):
        """Start the invoked task."""
        # create new db_task
        db_task = ProcessingTask(
            task_name="hinge-loss",
        )
        db_task.data["c"] = arguments.c
        db_task.save(commit=True)

        # include the database id in the callback data
        callback_data = ObjectiveFunctionInvokationCallbackSchema().dump(
            ObjectiveFunctionInvokationCallbackData(task_id=db_task.id)
        )

        make_callback(callback.callback_url, callback_data)


@HINGELOSS_BLP.route("/<int:db_id>/pass-data/")
class PassDataEndpoint(MethodView):
    """Endpoint to add additional data to the db task."""

    @HINGELOSS_BLP.arguments(
        ObjectiveFunctionPassDataSchema(unknown=EXCLUDE),
        location="json",
        required=True,
    )
    @HINGELOSS_BLP.response(HTTPStatus.OK, ObjectiveFunctionPassDataResponseSchema())
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
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

        db_task.save(commit=True)

        # the number of weights is the number of features
        return {"number_weights": input_data.x.shape[1]}


@HINGELOSS_BLP.route("/<int:db_id>/calc-loss-endpoint/")
class CalcLossEndpoint(MethodView):
    """Endpoint for the loss calculation."""

    @HINGELOSS_BLP.response(HTTPStatus.OK, LossResponseSchema())
    @HINGELOSS_BLP.arguments(
        CalcLossOrGradInputSchema(unknown=EXCLUDE), location="json", required=True
    )
    @HINGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: CalcLossOrGradInput, db_id: int) -> dict:
        """Endpoint for the calculation callback."""

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        # if x and y are not provided, use the ones from the db task
        if input_data.x is None:
            input_data.x = (
                SingleNumpyArraySchema().load({"array": db_task.data["x"]}).array
            )
        if input_data.y is None:
            input_data.y = (
                SingleNumpyArraySchema().load({"array": db_task.data["y"]}).array
            )

        loss = hinge_loss(
            X=input_data.x,
            y=input_data.y,
            w=input_data.x0,
            C=db_task.data["c"],
        )
        return {"loss": loss}
