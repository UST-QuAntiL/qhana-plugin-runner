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

import numpy as np
import torch
from flask import Response, redirect, render_template, request, url_for
from flask.globals import current_app
from flask.views import MethodView
from flask_smorest import abort
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import (
    ProcessingTask,
    TaskLink,
    TaskUpdateSubscription,
)
from qhana_plugin_runner.tasks import (
    TASK_STEPS_CHANGED,
    add_step,
    save_task_error,
    save_task_result,
)

from . import NN_BLP, NeuralNetwork
from .neural_network import NN
from .schemas import (
    CallbackUrl,
    CallbackUrlSchema,
    CombinedResponseSchema,
    EvaluateRequestSchema,
    EvaluateSchema,
    GradientResponseSchema,
    HyperparamterInputData,
    HyperparamterInputSchema,
    LossResponseSchema,
    PassDataSchema,
    WeightsResponseSchema,
)
from .tasks import clear_task_data, load_data, load_data_from_db


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
            links=[],
            entry_point=EntryPoint(
                href=url_for(f"{NN_BLP.name}.{SetupProcess.__name__}"),
                ui_href=url_for(
                    f"{NN_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
                ),
                plugin_dependencies=[],
                data_input=[],  # TODO: what about input data required in later steps? add a step(id) attribute to the metadata here?
                data_output=[
                    DataMetadata(  # FIXME
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

        # set default value
        if not data:
            data = {"number_of_neurons": 5}
        process_url = url_for(
            f"{NN_BLP.name}.{SetupProcess.__name__}",
            # forward the callback url to the processing step
            **callback_schema.dump(callback),
        )
        example_values_url = url_for(
            f"{NN_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
            **self.example_inputs,
            **callback_schema.dump(callback),
        )

        help_text = "The number of neurons in the hidden layer."

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


@NN_BLP.route("/hyperparameter/")
class SetupProcess(MethodView):
    """Save the hyperparameters to the database."""

    @NN_BLP.arguments(HyperparamterInputSchema(unknown=EXCLUDE), location="form")
    @NN_BLP.arguments(CallbackUrlSchema(unknown=EXCLUDE), location="query", required=True)
    @NN_BLP.response(HTTPStatus.SEE_OTHER)
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: HyperparamterInputData, callback: CallbackUrl):
        """Start the invoked task."""
        # create new db_task
        db_task = ProcessingTask(
            task_name="neural-network",
        )
        # save the data in the db
        db_task.data["number_of_neurons"] = arguments.number_of_neurons
        db_task.data["weights"] = -1

        db_task.save()
        DB.session.flush()

        # add callback as webhook subscriber subscribing to all updates
        subscription = TaskUpdateSubscription(
            db_task,
            webhook_href=callback.callback,
            task_href=url_for(
                "tasks-api.TaskView", task_id=str(db_task.id), _external=True
            ),
            event_type=None,
        )

        weights_link = TaskLink(
            db_task,
            type="of-weights",
            href=url_for(
                f"{NN_BLP.name}.{WeightsEndpoint.__name__}",
                db_id=db_task.id,
                _external=True,
            ),
        )
        calc_loss_link = TaskLink(
            db_task,
            type="of-evaluate",
            href=url_for(
                f"{NN_BLP.name}.{CalcLossEndpoint.__name__}",
                db_id=db_task.id,
                _external=True,
            ),
        )
        calc_grad_link = TaskLink(
            db_task,
            type="of-evaluate-gradient",
            href=url_for(
                f"{NN_BLP.name}.{CalcGradientEndpoint.__name__}",
                db_id=db_task.id,
                _external=True,
            ),
        )
        calc_loss_and_grad_link = TaskLink(
            db_task,
            type="of-evaluate-combined",
            href=url_for(
                f"{NN_BLP.name}.{CalcLossandGradEndpoint.__name__}",
                db_id=db_task.id,
                _external=True,
            ),
        )
        DB.session.add(weights_link)
        DB.session.add(calc_loss_link)
        DB.session.add(calc_grad_link)
        DB.session.add(calc_loss_and_grad_link)

        subscription.save()

        db_task.add_next_step(
            href=url_for(f"{NN_BLP.name}.{PassDataEndpoint.__name__}", db_id=db_task.id),
            ui_href=url_for(
                f"{NN_BLP.name}.{PassDataMicroFrontend.__name__}", db_id=db_task.id
            ),
            step_id="pass_data",
            commit=True,
        )

        app = current_app._get_current_object()
        TASK_STEPS_CHANGED.send(app, task_id=db_task.id)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@NN_BLP.route("/task/<int:db_id>/ui-pass-data/")
class PassDataMicroFrontend(MethodView):
    """Micro frontend for the pass_data step."""

    @NN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the pass_data step."
    )
    @NN_BLP.arguments(
        PassDataSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id=db_id)

    @NN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the pass_data step."
    )
    @NN_BLP.arguments(
        PassDataSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id=db_id)

    def render(self, data: Mapping, errors: dict, db_id: int):
        plugin = NeuralNetwork.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = PassDataSchema()

        process_url = url_for(f"{NN_BLP.name}.{PassDataEndpoint.__name__}", db_id=db_id)
        example_values_url = url_for(
            f"{NN_BLP.name}.{PassDataMicroFrontend.__name__}", db_id=db_id
        )

        return Response(
            render_template(
                "simple_template.html",
                name=NeuralNetwork.instance.name,
                version=NeuralNetwork.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text="Pass data to the objective function.",
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@NN_BLP.route("/task/<int:db_id>/pass-data/")
class PassDataEndpoint(MethodView):
    """Endpoint to load the features and target data."""

    @NN_BLP.arguments(
        PassDataSchema(unknown=EXCLUDE),
        location="form",
        required=True,
    )
    @NN_BLP.response(HTTPStatus.SEE_OTHER)
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: dict, db_id: int):
        """Load features and target data."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            abort(HTTPStatus.NOT_FOUND, message=msg)

        assert isinstance(db_task.data, dict)

        db_task.data["features_url"] = input_data["features"]
        db_task.data["target_url"] = input_data["target"]

        db_task.save(commit=True)

        task = load_data.s(db_id=db_id) | add_step.si(
            db_id=db_id,
            step_id="evaluate",
            href=url_for(f"{NN_BLP.name}.{EvaluateEndpoint.__name__}", db_id=db_id),
            ui_href=url_for(
                f"{NN_BLP.name}.{EvaluateMicroFrontend.__name__}", db_id=db_id
            ),
            task_log="Finished loading data.",
        )
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@NN_BLP.route("/task/<int:db_id>/ui-evaluate/")
class EvaluateMicroFrontend(MethodView):
    """Micro frontend for the evaluate step."""

    @NN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the evaluate step."
    )
    @NN_BLP.arguments(
        EvaluateSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id=db_id)

    @NN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the evaluate step."
    )
    @NN_BLP.arguments(
        EvaluateSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id=db_id)

    def render(self, data: Mapping, errors: dict, db_id: int):
        plugin = NeuralNetwork.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = EvaluateSchema()

        process_url = url_for(f"{NN_BLP.name}.{EvaluateEndpoint.__name__}", db_id=db_id)
        example_values_url = url_for(
            f"{NN_BLP.name}.{EvaluateMicroFrontend.__name__}", db_id=db_id
        )

        return Response(
            render_template(
                "simple_template.html",
                name=NeuralNetwork.instance.name,
                version=NeuralNetwork.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=process_url,  # URL of the processing step
                help_text="Complete the objective function task and clean up resources.",
                example_values=example_values_url,  # URL of this endpoint
            )
        )


@NN_BLP.route("/task/<int:db_id>/evaluate/")
class EvaluateEndpoint(MethodView):
    """Endpoint to complete the objective function task."""

    @NN_BLP.arguments(
        EvaluateSchema(unknown=EXCLUDE),
        location="json",
        required=True,
    )
    @NN_BLP.response(HTTPStatus.SEE_OTHER)
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: dict, db_id: int):
        """Complete the objective function task."""
        task = clear_task_data.s(db_id=db_id) | save_task_result.s(db_id=db_id)
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


#### Task Specific Endpoints ###################################################


@NN_BLP.route("/task/<int:db_id>/weights/")
class WeightsEndpoint(MethodView):
    """Endpoint for the number of weights."""

    @NN_BLP.response(HTTPStatus.OK, WeightsResponseSchema())
    @NN_BLP.require_jwt("jwt", optional=True)
    def get(self, db_id: int) -> dict:

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            abort(HTTPStatus.NOT_FOUND, message=msg)

        return {"weights": db_task.data.get("weights", -1)}


def _prepare_network(db_id: int, evaluate_input: dict):

    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        abort(HTTPStatus.NOT_FOUND, message=msg)

    assert isinstance(db_task.data, dict)

    weights = np.array(evaluate_input["weights"])
    features = load_data_from_db(db_task.data["features_key"])
    target = load_data_from_db(db_task.data["target_key"])
    number_of_neurons: int = db_task.data["number_of_neurons"]

    # create the neural network
    nn = NN(features.shape[1], number_of_neurons)
    # set the weights
    nn.set_weights(weights)

    return (
        nn,
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )


@NN_BLP.route("/task/<int:db_id>/loss/")
class CalcLossEndpoint(MethodView):
    """Endpoint for the loss calculation."""

    @NN_BLP.response(HTTPStatus.OK, LossResponseSchema())
    @NN_BLP.arguments(
        EvaluateRequestSchema(unknown=EXCLUDE), location="json", required=True
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: dict, db_id: int) -> dict:
        """Calculate the loss given the specific weights."""

        nn, features, target = _prepare_network(db_id=db_id, evaluate_input=input_data)

        loss = nn.get_loss(features, target)
        return {"loss": loss}


@NN_BLP.route("/<int:db_id>/calc-gradient-endpoint/")
class CalcGradientEndpoint(MethodView):
    """Endpoint for the gradient calculation."""

    @NN_BLP.response(HTTPStatus.OK, GradientResponseSchema())
    @NN_BLP.arguments(
        EvaluateRequestSchema(unknown=EXCLUDE), location="json", required=True
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: dict, db_id: int) -> dict:
        """Endpoint for the calculation callback."""

        nn, features, target = _prepare_network(db_id=db_id, evaluate_input=input_data)

        gradient = nn.get_gradient(features, target)
        return {"gradient": gradient.tolist()}


@NN_BLP.route("/<int:db_id>/calc-loss-and-grad/")
class CalcLossandGradEndpoint(MethodView):
    """Endpoint for the loss and gradient calculation."""

    @NN_BLP.response(HTTPStatus.OK, CombinedResponseSchema())
    @NN_BLP.arguments(
        EvaluateRequestSchema(unknown=EXCLUDE), location="json", required=True
    )
    @NN_BLP.require_jwt("jwt", optional=True)
    def post(self, input_data: dict, db_id: int) -> dict:
        """Endpoint for the calculation callback."""

        nn, features, target = _prepare_network(db_id=db_id, evaluate_input=input_data)

        loss, grad = nn.get_loss_and_gradient(features, target)
        return {"loss": loss, "gradient": grad.tolist()}
