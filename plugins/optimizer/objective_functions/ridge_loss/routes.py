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

from plugins.optimizer.coordinator.shared_schemas import (
    CalcLossInputData,
    CalcLossInputDataSchema,
    LossResponseSchema,
    ObjectiveFunctionCallbackData,
    ObjectiveFunctionCallbackSchema,
)
from plugins.optimizer.interaction_utils.schemas import CallbackURLData, CallbackURLSchema
from plugins.optimizer.interaction_utils.tasks import make_callback
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from . import RIDGELOSS_BLP, RidgeLoss
from .schemas import (
    HyperparamterInputData,
    HyperparamterInputSchema,
    RidgeLossTaskResponseSchema,
)
from .tasks import ridge_loss

TASK_LOGGER = get_task_logger(__name__)


@RIDGELOSS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @RIDGELOSS_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=RidgeLoss.instance.name,
            description=RidgeLoss.instance.description,
            name=RidgeLoss.instance.identifier,
            version=RidgeLoss.instance.version,
            type=PluginType.processing,
            tags=RidgeLoss.instance.tags,
            entry_point=EntryPoint(
                href=url_for(f"{RIDGELOSS_BLP.name}.{OptimizerCallbackProcess.__name__}"),
                ui_href=url_for(
                    f"{RIDGELOSS_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
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


@RIDGELOSS_BLP.route("/ui-hyperparameter/")
class HyperparameterSelectionMicroFrontend(MethodView):
    """Micro frontend for the hyperparameter selection."""

    example_inputs = {
        "alpha": 0.1,
    }

    @RIDGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @RIDGELOSS_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @RIDGELOSS_BLP.arguments(
        CallbackURLSchema(),
        location="query",
        required=False,
    )
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, callback: CallbackURLData):
        """Return the micro frontend."""
        return self.render(request.args, errors, callback)

    @RIDGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @RIDGELOSS_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @RIDGELOSS_BLP.arguments(
        CallbackURLSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, callback: CallbackURLData):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, callback)

    def render(self, data: Mapping, errors: dict, callback: CallbackURLData):
        plugin = RidgeLoss.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparamterInputSchema()
        callback_schema = CallbackURLSchema()

        if not data:
            data = {"alpha": 0.1}
        process_url = url_for(
            f"{RIDGELOSS_BLP.name}.{OptimizerCallbackProcess.__name__}",
            **callback_schema.dump(callback),
        )
        example_values_url = url_for(
            f"{RIDGELOSS_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
            **self.example_inputs,
            **callback_schema.dump(callback),
        )

        help_text = r"""Alpha is the regularization strength. Larger values specify stronger regularization.
        The formula for ridge loss with regularization strength alpha is:
        \$\$\sum_{i=1}^n (y_i - w^T x_i)^2 + \alpha ||w||_2^2\$\$"""

        return Response(
            render_template(
                "simple_template.html",
                name=RidgeLoss.instance.name,
                version=RidgeLoss.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text=help_text,
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@RIDGELOSS_BLP.route("/optimizer-callback/")
class OptimizerCallbackProcess(MethodView):
    """Make a callback to the optimizer plugin after selection the hyperparameter."""

    @RIDGELOSS_BLP.arguments(HyperparamterInputSchema(unknown=EXCLUDE), location="form")
    @RIDGELOSS_BLP.arguments(
        CallbackURLSchema(unknown=EXCLUDE), location="query", required=True
    )
    @RIDGELOSS_BLP.response(HTTPStatus.OK, RidgeLossTaskResponseSchema())
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: HyperparamterInputData, callback: CallbackURLData):
        """Start the invoked task."""
        # create new db_task
        db_task = ProcessingTask(
            task_name="ridge-loss",
        )
        db_task.data["alpha"] = arguments.alpha
        db_task.save(commit=True)
        callback_url = callback.callback_url
        calc_endpoint_url = url_for(
            f"{RIDGELOSS_BLP.name}.{CalcCallbackEndpoint.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        callback_schema = ObjectiveFunctionCallbackSchema()
        callback_data = callback_schema.dump(
            ObjectiveFunctionCallbackData(calc_loss_endpoint_url=calc_endpoint_url)
        )

        make_callback(callback_url, callback_data)


@RIDGELOSS_BLP.route("/<int:db_id>/calc-callback-endpoint/")
class CalcCallbackEndpoint(MethodView):
    """Endpoint for the calculation callback."""

    @RIDGELOSS_BLP.response(HTTPStatus.OK, LossResponseSchema())
    @RIDGELOSS_BLP.arguments(
        CalcLossInputDataSchema(unknown=EXCLUDE), location="json", required=True
    )
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: CalcLossInputData, db_id: int) -> dict:
        """Endpoint for the calculation callback."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        loss = ridge_loss(
            X=input_data.x,
            y=input_data.y,
            w=input_data.x0,
            alpha=db_task.data["alpha"],
        )
        return {"loss": loss}
