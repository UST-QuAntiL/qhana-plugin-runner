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

_plugin_name = "state-preparation"
__version__ = "v0.1"
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
    # preparation_method = ma.fields.Integer(
    #     required=True,
    #     allow_none=None,
    #     metadata={
    #         "label": "Method",
    #         "description": "BASIS_ENCODING = 1    ANGLE_ENCODING = 2    ARBITRARY_STATE = 3",
    #         "input_type": "number",
    #     },
    # )


    data_values = CSVList(
        required=False,
        allow_none=True,
        element_type=ma.fields.String,
        metadata={
            "label": "Data",
            "description": "Data to be encoded; comma seperated.",
            "input_type": "textarea",
        },
    )
    number_of_qbits = ma.fields.Integer(
        required=False,
        allow_none=True,
        metadata={
            "label": "Number of Qubits",
            "description": "Number of qubits for the state preparation.",
            "input_type": "number",
        },
    )
    most_significant_value = ma.fields.Integer(
        required=False,
        allow_none=True,
        metadata={
            "label": "Value of most significant bit",
            "description": "The value of the most significant bit",
            "input_type": "number",
        },
    )

    append_measurement = ma.fields.Bool(
        required=False,
        missing=False,
        metadata={
            "label": "Measurement",
            "description" : "Append a measurement to the circuit.",
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
                        name="state_preparation.qasm",
                    ),
                ],
            ),
            tags=StatePreparation.instance.tags,
        )


@STATE_PREPARATION_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the state preparation plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
        "method": METHODENUM.BASIS_ENCODING,
        "data_values": "0,1,0,1",
        "number_of_qbits": 2,
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
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

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
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{STATE_PREPARATION_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@STATE_PREPARATION_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @STATE_PREPARATION_BLP.arguments(StatePreparationParametersSchema(unknown=EXCLUDE), location="form")
    @STATE_PREPARATION_BLP.response(HTTPStatus.SEE_OTHER)
    @STATE_PREPARATION_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(task_name=prepare_task.name, parameters=StatePreparationParametersSchema().dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = prepare_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
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
    description = "Tests the connection of all components by printing some text (UPDATED!)."
    tags = ["state-preparation", "demo"]

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
    
    method: Optional[int] = StatePreparationParametersSchema().loads(task_data.parameters or "{}").get("preparation_method", None)
    data_values: Optional[str] =  StatePreparationParametersSchema().loads(task_data.parameters or "{}").get("data_values", None)
    number_of_qbits: Optional[int] =  StatePreparationParametersSchema().loads(task_data.parameters or "{}").get("number_of_qbits", None)
    most_significant_value: Optional[int] =  StatePreparationParametersSchema().loads(task_data.parameters or "{}").get("most_significant_value", None)
    append_measurement: Optional[bool] =  StatePreparationParametersSchema().loads(task_data.parameters or "{}").get("append_measurement", None)

    TASK_LOGGER.info(f"Loaded input parameters from db: data_values='{data_values}', number_of_qbits='{number_of_qbits}', most_significant_value='{most_significant_value}', append_measurement='{append_measurement}'")



    #TODO instead get number of digits before and after decimal point from input directly
    if method == METHODENUM.BASIS_ENCODING:
        if len(data_values) != 1:
            raise ValueError("Basis encoding requires exactly one data value!")
        value=data_values[0]

        if (most_significant_value & (most_significant_value - 1)) != 0:
            raise ValueError("The most significant bit's value must be a power of 2!")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                raise ValueError("The data value for basis encoding must be a number!")
        
        if value < 0 or value >= 2*most_significant_value:
            raise ValueError(f"The data value for basis encoding must be in the range [0,{2*most_significant_value})!")
        
        bit_list=[]
        for i in range(number_of_qbits):
            cur_bit_value=most_significant_value/(2**i)
            if value >= cur_bit_value:
                value -= cur_bit_value
                bit_list.append(1)
            else:
                bit_list.append(0)
            

        qc = QuantumCircuit(number_of_qbits)
        for i, bit in enumerate(reversed(bit_list)):
            if bit == 1:
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
        normalized_values = [1/2 * np.pi * (x - xmin) / (xmax - xmin) for x in data_values]

        qc = QuantumCircuit(len(data_values))
        for i, value in enumerate(normalized_values):
            qc.ry(2*value, i)

        
    elif method == METHODENUM.ARBITRARY_STATE:
        try:
            data_values = [complex(x) for x in data_values]
        except ValueError:
            raise ValueError("Data values must be complex numbers!")
        qc=StatePreparationQiskit(data_values)
    else:
        raise NotImplementedError(f"Method {method} not implemented!")
    
    if append_measurement:
        qc.measure_all()
    qasm_str=qasm3.dumps(qc)
        


    with SpooledTemporaryFile(mode="w") as output:
        output.write(qasm_str)
        STORE.persist_task_result(
            db_id, output, "state_preparation.qasm", "executable/circuit", "text/x-qasm"
        )
    return qasm_str
