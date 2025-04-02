# Copyright 2021 QHAna plugin runner contributors.
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
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional
from enum import Enum

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import abort, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from qhana_plugin_runner.api.extra_fields import EnumField, CSVList

# Optimizer specfic packages
# from scipy.optimize import minimize

_plugin_name = "optimizer"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


OPTIMIZER_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Optimizer plugin API.",
    template_folder="optimizer_templates",
)


class OPTIMIZERENUM(Enum):
    SPSA = "SPSA (Simultaneous Perturbation Stochastic Approximation)"
    COBYLA = "COBYLA (Constrained Optimization BY Linear Approximation)"


# Stuff defined here shows up directly in Frontend
class OptimizerParametersSchema(FrontendFormBaseSchema):
    optimizer = EnumField(
        OPTIMIZERENUM,
        required=False,
        allow_none=False,
        metadata={
            "label": "Optimizer",
            "description": "Select optimizer.",
            "input_type": "select",
        },
    )

    # Parameters to optimize
    # TODO check if there is a better dtype for this
    params = CSVList(
        required=True,  # TODO Check if True works
        allow_none=False,  # TODO Check if False works
        element_type=ma.fields.String,
        metadata={
            "label": "Parameters",
            "description": "List of ansatz parameters to optimize; comma separated.",
            "input_type": "textarea",
        },
    )

    cost_function_result = ma.fields.Float(
        required=False,
        allow_none=False,
        metadata={
            "label": "Cost function result",
            "description": "Result of the cost function on trainings data.",
            "input_type": "number",  # check if this should be defined to float
        },
    )


@OPTIMIZER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @OPTIMIZER_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = Optimizer.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{OPTIMIZER_BLP.name}.ProcessView"),
                ui_href=url_for(f"{OPTIMIZER_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="executable/circuit",
                        content_type=["text/x-qasm"],
                        required=True,
                    ),
                ],
            ),
            tags=Optimizer.instance.tags,
        )


@OPTIMIZER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the optimizer plugin."""

    # TODO update to new fields
    example_inputs = {
        "optimizer": OPTIMIZERENUM.SPSA,
        "params": "0,2,2,2,2",
        "cost_function_result": 6,
    }

    @OPTIMIZER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the optimizer plugin."
    )
    @OPTIMIZER_BLP.arguments(
        OptimizerParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @OPTIMIZER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the optimizer plugin."
    )
    @OPTIMIZER_BLP.arguments(
        OptimizerParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = Optimizer.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = OptimizerParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{OPTIMIZER_BLP.name}.ProcessView"),
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OPTIMIZER_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@OPTIMIZER_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @OPTIMIZER_BLP.arguments(
        OptimizerParametersSchema(unknown=EXCLUDE), location="form"
    )
    @OPTIMIZER_BLP.response(HTTPStatus.SEE_OTHER)
    @OPTIMIZER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name=prepare_task.name,
            parameters=OptimizerParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = prepare_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class Optimizer(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    # TODO check what needs to be done s.t. these update in UI
    description = "Optimize parameters based on calculated cost."
    tags = ["optimizer", "qnn"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return OPTIMIZER_BLP

    def get_requirements(self) -> str:
        # return "qiskit~=1.3.2\nnumpy"
        return "scipy"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Optimizer.instance.identifier}.prepare_task", bind=True)
def prepare_task(self, db_id: int) -> str:
    import numpy as np
    from qiskit import QuantumCircuit, qasm3
    from qiskit.circuit.library import StatePreparation as StatePreparationQiskit

    TASK_LOGGER.info(f"Starting new prepare task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    optimizer: int = (
        OptimizerParametersSchema()
        .loads(task_data.parameters or "{}")
        .get("optimizer", None)
    )
    params: list[str] = (
        OptimizerParametersSchema()
        .loads(task_data.parameters or "{}")
        .get("params", None)
    )
    cost_function_result: float = (
        OptimizerParametersSchema()
        .loads(task_data.parameters or "{}")
        .get("cost_function_result", None)
    )

    TASK_LOGGER.info(
        f"Loaded input parameters from db: params='{params}', cost_function_result='{cost_function_result}'"
    )
    
    return "just testing"

    # # Stuff required for every method
    # try:
    #     params = [float(param) for param in params]
    # except ValueError:
    #     raise ValueError("Data values must be numbers!")

    # if optimizer == OPTIMIZERENUM.SPSA:
    #     if len(data_values) != 1:
    #         raise ValueError("Basis encoding requires exactly one data value!")
    #     value = data_values[0]

    #     try:
    #         value = float(value)
    #     except ValueError:
    #         raise ValueError("The data value for basis encoding must be a number!")

    #     integer_part = int(value)
    #     if integer_part > 2 ** digits_before_decimal - 1:
    #         raise ValueError(f"Integer part {integer_part} is too large for {digits_before_decimal} bits!")
    #     fractional_part = value - integer_part

    #     integer_bits = bin(integer_part)[2:].zfill(digits_before_decimal)
    #     fractional_bits = bin(int(fractional_part * (2 ** digits_after_decimal)))[2:].zfill(digits_after_decimal)

    #     bit_list = sign_bit + integer_bits + fractional_bits

    #     qc = QuantumCircuit(len(bit_list))
    #     for i, bit in enumerate(reversed(bit_list)):
    #         if bit == '1':
    #             qc.x(i)

    # elif optimizer == OPTIMIZERENUM.COBYLA:

    #     xmin = min(data_values)
    #     xmax = max(data_values)
    #     if (xmax - xmin) == 0:
    #         raise ValueError("Data values must not be all the same!")
    #     normalized_values = [1/2 * np.pi * (x - xmin) / (xmax - xmin) for x in data_values]

    #     qc = QuantumCircuit(len(data_values))
    #     for i, value in enumerate(normalized_values):
    #         qc.ry(2*value, i)

    # elif optimizer == OPTIMIZERENUM.ARBITRARY_STATE:
    #     try:
    #         data_values = [complex(x) for x in data_values]
    #     except ValueError:
    #         raise ValueError("Data values must be complex numbers!")
    #     qc=StatePreparationQiskit(data_values)
    # else:
    #     raise NotImplementedError(f"Method {optimizer} not implemented!")

    # # TODO find return type
    # # TODO adapt to optimizer
    # with SpooledTemporaryFile(mode="w") as output:
    #     output.write(qasm_str)
    #     STORE.persist_task_result(
    #         db_id, output, "state_preparation.qasm", "executable/circuit", "text/x-qasm"
    #     )
    # return qasm_str
