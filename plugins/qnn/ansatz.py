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

_plugin_name = "ansatz"
__version__ = "v0.1"
_identifier = plugin_identifier(_plugin_name, __version__)


ANSATZ_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Ansatz plugin API.",
    template_folder="ansatz_templates",
)

class ENTANGLEMENT(Enum):
    REVERSE_LINEAR = "reverse_linear"
    LINEAR = "linear"
    FULL = "full"
    CIRCULAR = "circular"

class ANSATZMETHOD(Enum):
    REAL_AMPLITUDES = "real_amplitudes"
    EFFICIENT_SU2 = "efficient_su2"



class AnsatzParametersSchema(FrontendFormBaseSchema):
    ansatzmethod = EnumField(
        ANSATZMETHOD,
        required=True,
        allow_none=False,
        metadata={
            "label": "Ansatz Method",
            "description": "Select the ansatz method.",
            "input_type": "select",
        },
    )
    entanglement = EnumField(
        ENTANGLEMENT,
        required=True,
        allow_none=False,
        metadata={
            "label": "Entanglement",
            "description": "Select the entanglement strategy.",
            "input_type": "select",
        },
    )
    num_qubits = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Qubits",
            "description": "Number of qubits for the ansatz.",
            "input_type": "number",
        },
    )
    num_layers = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Layers",
            "description": "Number of layers for the ansatz.",
            "input_type": "number",
        },
    )
    append_measurement = ma.fields.Bool(
        required=False,
        missing=False,
        metadata={
            "label": "Measurement",
            "description": "Append a measurement to the circuit.",
        },
    )


@ANSATZ_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @ANSATZ_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @ANSATZ_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = Ansatz.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{ANSATZ_BLP.name}.ProcessView"),
                ui_href=url_for(f"{ANSATZ_BLP.name}.MicroFrontend"),
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
            tags=Ansatz.instance.tags,
        )


@ANSATZ_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the ansatz plugin."""

    example_inputs = {
        "entanglement": ENTANGLEMENT.LINEAR,
        "num_qubits": 3,
        "append_measurement": True,
    }

    @ANSATZ_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the ansatz plugin."
    )
    @ANSATZ_BLP.arguments(
        AnsatzParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @ANSATZ_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @ANSATZ_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the ansatz plugin."
    )
    @ANSATZ_BLP.arguments(
        AnsatzParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @ANSATZ_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = Ansatz.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = AnsatzParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{ANSATZ_BLP.name}.ProcessView"),
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{ANSATZ_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@ANSATZ_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @ANSATZ_BLP.arguments(AnsatzParametersSchema(unknown=EXCLUDE), location="form")
    @ANSATZ_BLP.response(HTTPStatus.SEE_OTHER)
    @ANSATZ_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(task_name=ansatz_task.name, parameters=AnsatzParametersSchema().dumps(arguments))
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = ansatz_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class Ansatz(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Tests the connection of all components by printing some text (UPDATED!)."
    tags = ["ansatz", "demo"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return ANSATZ_BLP
    
    def get_requirements(self) -> str:
        return "qiskit~=1.3.2\nnumpy"


TASK_LOGGER = get_task_logger(__name__)

@CELERY.task(name=f"{Ansatz.instance.identifier}.ansatz_task", bind=True)
def ansatz_task(self, db_id: int) -> str:
    from qiskit import qasm3
    from qiskit.circuit.library import RealAmplitudes, EfficientSU2

    TASK_LOGGER.info(f"Starting new prepare task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)
    
    ansatzmethod: Optional[str] = AnsatzParametersSchema().loads(task_data.parameters or "{}").get("ansatzmethod", None)
    entanglement: Optional[str] = AnsatzParametersSchema().loads(task_data.parameters or "{}").get("entanglement", None)
    num_qubits: Optional[int] = AnsatzParametersSchema().loads(task_data.parameters or "{}").get("num_qubits", None)
    num_layers: Optional[int] = AnsatzParametersSchema().loads(task_data.parameters or "{}").get("num_layers", None)
    append_measurement: Optional[bool] = AnsatzParametersSchema().loads(task_data.parameters or "{}").get("append_measurement", None)


    TASK_LOGGER.info(f"Loaded input parameters from db: ansatzmethod='{ansatzmethod}', entanglement='{entanglement}', num_qubits='{num_qubits}', num_layers='{num_layers}', append_measurement='{append_measurement}'")

    if ansatzmethod == ANSATZMETHOD.REAL_AMPLITUDES:
        ansatz = RealAmplitudes(num_qubits=num_qubits, entanglement=entanglement.value,reps=num_layers,parameter_prefix='p')

        if append_measurement:
            ansatz.measure_all()

        qasm_str = qasm3.dumps(ansatz)
    
    elif ansatzmethod == ANSATZMETHOD.EFFICIENT_SU2:
        ansatz = EfficientSU2(num_qubits=num_qubits, entanglement=entanglement.value,reps=num_layers,parameter_prefix='p')

        if append_measurement:
            ansatz.measure_all()

        qasm_str = qasm3.dumps(ansatz)

    with SpooledTemporaryFile(mode="w") as output:
        output.write(qasm_str)
        STORE.persist_task_result(
            db_id, output, "ansatz.qasm", "executable/circuit", "text/x-qasm"
        )
    return qasm_str
