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
from tempfile import SpooledTemporaryFile
from typing import Mapping, Optional
from enum import Enum
from typing import Any, ChainMap

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
from qhana_plugin_runner.api.extra_fields import EnumField

_plugin_name = "state-preparation"
__version__ = "v0.2.0"
_identifier = plugin_identifier(_plugin_name, __version__)


STATE_PREPARATION_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="State preparation plugin API.",
    template_folder="state_preparation_templates",
)


class METHODENUM(Enum):
    BASIS_ENCODING = "Basis Encoding"
    ANGLE_ENCODING = "Angle Encoding"
    ARBITRARY_STATE = "Arbitrary State"


class StatePreparationParametersSchema(FrontendFormBaseSchema):
    preparation_method = EnumField(
        METHODENUM,
        required=True,
        allow_none=False,
        metadata={
            "label": "Method",
            "description": "Select the state preparation method.",
            "input_type": "select",
        },
    )

    data_values = ma.fields.String(
        required=False,
        allow_none=True,
        element_type=ma.fields.String,
        metadata={
            "label": "Data",
            "description": "Single value, or list; for complex numbers use [[real, imag], [real, imag], ...]",
            "input_type": "textarea",
        },
    )
    digits_before_decimal = ma.fields.Integer(
        required=False,
        allow_none=True,
        metadata={
            "label": "Digits Before Decimal",
            "description": "Number of digits before the decimal point (only needed for Basis Encoding).",
            "input_type": "number",
        },
    )
    digits_after_decimal = ma.fields.Integer(
        required=False,
        allow_none=True,
        metadata={
            "label": "Digits After Decimal",
            "description": "Number of digits after the decimal point (only needed for Basis Encoding).",
            "input_type": "number",
        },
    )

    encode_sign = ma.fields.Bool(
        required=False,
        missing=False,
        metadata={
            "label": "Encode Sign",
            "description": "Use the first bit to encode positive/negative values.",
        },
    )


@STATE_PREPARATION_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @STATE_PREPARATION_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @STATE_PREPARATION_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = StatePreparation.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{STATE_PREPARATION_BLP.name}.ProcessView"),
                ui_href=url_for(f"{STATE_PREPARATION_BLP.name}.MicroFrontend"),
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
            tags=StatePreparation.instance.tags,
        )


@STATE_PREPARATION_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the state preparation plugin."""

    example_inputs = {}

    default_inputs = {
        "digitsBeforeDecimal": 0,
        "digitsAfterDecimal": 0,
    }

    @STATE_PREPARATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the state preparation plugin."
    )
    @STATE_PREPARATION_BLP.arguments(
        StatePreparationParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @STATE_PREPARATION_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        values: ChainMap[str, Any] = ChainMap(request.args.to_dict(), self.default_inputs)
        return self.render(values, errors, False)

    @STATE_PREPARATION_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the state preparation plugin."
    )
    @STATE_PREPARATION_BLP.arguments(
        StatePreparationParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @STATE_PREPARATION_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = StatePreparation.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = StatePreparationParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{STATE_PREPARATION_BLP.name}.ProcessView"),
                help_text="",
                example_values=url_for(
                    f"{STATE_PREPARATION_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@STATE_PREPARATION_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @STATE_PREPARATION_BLP.arguments(
        StatePreparationParametersSchema(unknown=EXCLUDE), location="form"
    )
    @STATE_PREPARATION_BLP.response(HTTPStatus.SEE_OTHER)
    @STATE_PREPARATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the state_preparation task."""
        db_task = ProcessingTask(
            task_name=prepare_task.name,
            parameters=StatePreparationParametersSchema().dumps(arguments),
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


class StatePreparation(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Plugin for state preparation in quantum circuits."
    tags = ["state-preparation", "encoding", "qnn", "qiskit-1.3.2", "qasm-3"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return STATE_PREPARATION_BLP

    def get_requirements(self) -> str:
        return "qiskit~=1.3.2\nnumpy"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{StatePreparation.instance.identifier}.prepare_task", bind=True)
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

    method: Optional[int] = (
        StatePreparationParametersSchema()
        .loads(task_data.parameters or "{}")
        .get("preparation_method", None)
    )
    data_values: Optional[str] = (
        StatePreparationParametersSchema()
        .loads(task_data.parameters or "{}")
        .get("data_values", None)
    )
    digits_before_decimal: Optional[int] = (
        StatePreparationParametersSchema()
        .loads(task_data.parameters or "{}")
        .get("digits_before_decimal", None)
    )
    digits_after_decimal: Optional[int] = (
        StatePreparationParametersSchema()
        .loads(task_data.parameters or "{}")
        .get("digits_after_decimal", None)
    )
    encode_sign: Optional[bool] = (
        StatePreparationParametersSchema()
        .loads(task_data.parameters or "{}")
        .get("encode_sign", None)
    )

    TASK_LOGGER.info(
        f"Loaded input parameters from db: data_values='{data_values}', digits_before_decimal='{digits_before_decimal}', digits_after_decimal='{digits_after_decimal}', encode_sign='{encode_sign}'"
    )

    # add brackets if not present
    if not data_values[0] == "[":
        data_values = "[" + data_values + "]"
    # convert string input to list
    try:
        data_values = eval(data_values)
    except Exception as e:
        TASK_LOGGER.error(f"Could not parse data values '{data_values}'! Error: {e}")
        raise ValueError(f"Could not parse data values '{data_values}'! Error: {e}")
    if method == METHODENUM.BASIS_ENCODING:
        if len(data_values) != 1:
            raise ValueError("Basis encoding requires exactly one data value!")
        value = data_values[0]

        try:
            value = float(value)
        except ValueError:
            raise ValueError("The data value for basis encoding must be a number!")

        if encode_sign:
            if value < 0:
                value = -value
                sign_bit = "1"
            else:
                sign_bit = "0"
        else:
            if value < 0:
                raise ValueError("Negative values are not allowed without sign encoding!")
            sign_bit = ""

        integer_part = int(value)
        if integer_part > 2**digits_before_decimal - 1:
            raise ValueError(
                f"Integer part {integer_part} is too large for {digits_before_decimal} bits!"
            )
        fractional_part = value - integer_part

        integer_bits = bin(integer_part)[2:].zfill(digits_before_decimal)
        fractional_bits = bin(int(fractional_part * (2**digits_after_decimal)))[2:]
        fractional_bits = fractional_bits.ljust(digits_after_decimal, "0")

        bit_list = sign_bit + integer_bits + fractional_bits

        qc = QuantumCircuit(len(bit_list))
        for i, bit in enumerate(bit_list):
            if bit == "1":
                qc.x(i)

    elif method == METHODENUM.ANGLE_ENCODING:
        try:
            data_values = [float(x) for x in data_values]
        except ValueError:
            raise ValueError("Data values must be numbers!")

        xmin = min(data_values)
        xmax = max(data_values)
        if (xmax - xmin) == 0:
            raise ValueError("Data values must not be all the same!")
        normalized_values = [
            1 / 2 * np.pi * (x - xmin) / (xmax - xmin) for x in data_values
        ]

        qc = QuantumCircuit(len(data_values))
        for i, value in enumerate(normalized_values):
            qc.ry(2 * value, i)

    elif method == METHODENUM.ARBITRARY_STATE:
        try:
            data_values = [complex(real, imag) for real, imag in data_values]
        except ValueError:
            raise ValueError("Data values must be complex numbers!")
        gate = StatePreparationQiskit(data_values)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, range(gate.num_qubits))
        # neccessary for arbitrary state as qasm import does not support complex numbers
        qc = qc.decompose()
    else:
        raise NotImplementedError(f"Method {method} not implemented!")

    qasm_str = qasm3.dumps(qc)

    with SpooledTemporaryFile(mode="w") as output:
        output.write(qasm_str)
        STORE.persist_task_result(
            db_id, output, "state_preparation.qasm", "executable/circuit", "text/x-qasm"
        )
    return qasm_str
