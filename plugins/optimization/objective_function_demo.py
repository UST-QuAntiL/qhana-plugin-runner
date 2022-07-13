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
import json
from http import HTTPStatus
from json import dumps
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional, Dict, List

import marshmallow as ma
import numpy as np
import requests
import torch
from celery import chain
from celery.utils.log import get_task_logger
from flask import Response, abort, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from torch import nn

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    InteractionEndpoint,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result, add_step
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "objective-function-demo"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


OBJ_FUNC_DEMO_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="API of objective function plugin that can be used by other plugins.",
    template_folder="hello_world_templates",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class HyperparametersSchema(FrontendFormBaseSchema):
    number_of_input_values = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of input values.",
            "description": "Number of input values for the neural network.",
            "input_type": "text",
        },
    )
    number_of_neurons = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of neurons",
            "description": "Number of neurons for the neural network.",
            "input_type": "text",
        },
    )


class CalculationInputSchema(MaBaseSchema):
    data_set = ma.fields.Url(required=True, allow_none=False)
    hyperparameters = ma.fields.Url(required=True, allow_none=False)
    parameters = ma.fields.List(ma.fields.Float(), required=True, allow_none=False)


class CalculationOutputSchema(MaBaseSchema):
    objective_value = ma.fields.Float(required=True, allow_none=False)


@OBJ_FUNC_DEMO_BLP.route("/", defaults={"db_id": 0})
@OBJ_FUNC_DEMO_BLP.route("/<int:db_id>/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @OBJ_FUNC_DEMO_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, db_id):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=ObjectiveFunctionDemo.instance.name,
            description=OBJ_FUNC_DEMO_BLP.description,
            name=ObjectiveFunctionDemo.instance.identifier,
            version=ObjectiveFunctionDemo.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{OBJ_FUNC_DEMO_BLP.name}.SetupView", db_id=db_id),
                ui_href=url_for(f"{OBJ_FUNC_DEMO_BLP.name}.MicroFrontend", db_id=db_id),
                interaction_endpoints=[
                    InteractionEndpoint(
                        type="objective-function-calculation",
                        href=url_for(f"{OBJ_FUNC_DEMO_BLP.name}.CalculationView"),
                    )
                ],
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
            tags=["objective-function"],
        )


@OBJ_FUNC_DEMO_BLP.route("/<int:db_id>/ui/")
class MicroFrontend(MethodView):
    """Micro frontend of the objective function demo plugin."""

    @OBJ_FUNC_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the objective function demo plugin."
    )
    @OBJ_FUNC_DEMO_BLP.arguments(
        HyperparametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id)

    @OBJ_FUNC_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the objective function demo plugin."
    )
    @OBJ_FUNC_DEMO_BLP.arguments(
        HyperparametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id)

    def render(self, data: Mapping, errors: dict, db_id: int):
        plugin = ObjectiveFunctionDemo.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparametersSchema()

        return Response(
            render_template(
                "simple_template.html",
                name=ObjectiveFunctionDemo.instance.name,
                version=ObjectiveFunctionDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OBJ_FUNC_DEMO_BLP.name}.SetupView", db_id=db_id
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OBJ_FUNC_DEMO_BLP.name}.MicroFrontend",
                    db_id=db_id,
                ),  # URL of this endpoint
            )
        )


@OBJ_FUNC_DEMO_BLP.route("/<int:db_id>/setup/")
class SetupView(MethodView):
    """Start the setup task."""

    @OBJ_FUNC_DEMO_BLP.arguments(HyperparametersSchema(unknown=EXCLUDE), location="form")
    @OBJ_FUNC_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the setup task."""
        optimizer_db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if optimizer_db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)
        optimizer_db_task.clear_previous_step()
        optimizer_db_task.parameters = dumps(arguments)
        optimizer_db_task.save(commit=True)

        # add the next processing step with the data that was stored by the previous step of the invoking plugin
        task: chain = setup_task.s(db_id=optimizer_db_task.id) | add_step.s(
            db_id=optimizer_db_task.id,
            step_id=optimizer_db_task.data["next_step_id"],  # name of the next sub-step
            href=optimizer_db_task.data[
                "href"
            ],  # URL to the processing endpoint of the next step
            ui_href=optimizer_db_task.data[
                "ui_href"
            ],  # URL to the micro frontend of the next step
            prog_value=66,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=optimizer_db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(optimizer_db_task.id)),
            HTTPStatus.SEE_OTHER,
        )


class ObjectiveFunctionDemo(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return OBJ_FUNC_DEMO_BLP

    def get_requirements(self) -> str:
        return "scipy~=1.8.1\ntorch~=1.12"


TASK_LOGGER = get_task_logger(__name__)


class NeuralNetwork(nn.Module):
    def __init__(self, number_of_input_values: int, number_of_hidden_units: int):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(number_of_input_values, number_of_hidden_units),
            nn.ReLU(),
            nn.Linear(number_of_hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

    def get_loss(self, input_data: torch.Tensor, target_data: torch.Tensor):
        with torch.no_grad():
            output = self(input_data)

            loss = torch.mean((output - target_data) * (output - target_data))

        return loss.item()

    def get_param_list(self) -> List[float]:
        params = list(self.parameters())
        param_list = []

        for param in params:
            param_list.extend(param.data.detach().flatten().tolist())

        return param_list

    def set_param_list(self, params: List[float]):
        index = 0

        for param in self.parameters():
            number_of_elements = param.data.numel()
            param.data = torch.tensor(
                params[index : index + number_of_elements], dtype=torch.float32
            ).resize_as_(param.data)
            index += number_of_elements

    def set_param_get_loss(
        self, x: np.ndarray, input_data: torch.Tensor, target_data: torch.Tensor
    ):
        self.set_param_list(list(x))

        return self.get_loss(input_data, target_data)

    def get_number_of_parameters(self):
        return len(self.get_param_list())


@CELERY.task(name=f"{ObjectiveFunctionDemo.instance.identifier}.setup_task", bind=True)
def setup_task(self, db_id: int) -> str:
    """
    Retrieves the input data from the database and stores metadata and hyperparameters into files.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting setup task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    parameters: Dict = json.loads(task_data.parameters)
    number_of_input_values: int = parameters.get("number_of_input_values")
    number_of_neurons: int = parameters.get("number_of_neurons")

    TASK_LOGGER.info(
        f"Loaded data from db: number_of_input_values='{number_of_input_values}'"
    )
    TASK_LOGGER.info(f"Loaded data from db: number_of_neurons='{number_of_neurons}'")

    if number_of_input_values is None or number_of_neurons is None:
        raise ValueError("Input parameters incomplete")

    model = NeuralNetwork(number_of_input_values, number_of_neurons)

    metadata = {"number_of_parameters": model.get_number_of_parameters()}

    with SpooledTemporaryFile(mode="w") as output:
        output.write(dumps(metadata))
        STORE.persist_task_result(
            db_id,
            output,
            "metadata.json",
            "objective-function-metadata",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        output.write(task_data.parameters)
        STORE.persist_task_result(
            db_id,
            output,
            "hyperparameters.json",
            "objective-function-demo-hyperparameters",
            "application/json",
        )

    return "Stored metadata and hyperparameters"


@OBJ_FUNC_DEMO_BLP.route("/calc/")
class CalculationView(MethodView):
    """Objective function calculation resource."""

    @OBJ_FUNC_DEMO_BLP.arguments(CalculationInputSchema())
    @OBJ_FUNC_DEMO_BLP.response(HTTPStatus.OK, CalculationOutputSchema())
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Endpoint calculating the objective value."""
        data_set_url: str = arguments["data_set"]
        hyperparameters_url: str = arguments["hyperparameters"]
        parameters: List[float] = arguments["parameters"]

        data_set = requests.get(data_set_url).json()
        hyperparameters = requests.get(hyperparameters_url).json()

        model = NeuralNetwork(
            hyperparameters["number_of_input_values"],
            hyperparameters["number_of_neurons"],
        )
        model.set_param_list(parameters)
        loss = model.get_loss(
            torch.tensor(data_set["input"], dtype=torch.float32),
            torch.tensor(data_set["target"], dtype=torch.float32),
        )

        return {"objective_value": loss}
