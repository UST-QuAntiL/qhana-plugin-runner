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

import mimetypes
import re
import time
from collections import ChainMap
from http import HTTPStatus
from json import dump, dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, Mapping, Optional, Union, cast
from uuid import uuid4

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
from marshmallow.validate import Range

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    InputDataMetadata,
    OutputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "AmazonBraket_LocalSimulator"
__version__ = "v1.0.0"
_identifier = plugin_identifier(_plugin_name, __version__)


BRAKET_LOCAL_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Circuit executor exposing the braket_local simulators as backend.",
)


class BraketSimulatorParametersSchema(FrontendFormBaseSchema):
    circuit = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="executable/circuit",
        data_content_types="text/x-qasm",
        metadata={
            "label": "OpenQASM 3 Circuit",
            "description": "URL to a quantum circuit in the OpenQASM format.",
            "input_type": "text",
        },
    )
    executionOptions = FileUrl(
        required=False,
        allow_none=True,
        load_missing=None,
        data_input_type="provenance/execution-options",
        data_content_types=["text/csv", "application/json", "application/X-lines+json"],
        metadata={
            "label": "Execution Options (optional)",
            "description": "URL to a file containing execution options. (optional)",
            "input_type": "text",
        },
    )
    shots = ma.fields.Integer(
        required=False,
        allow_none=True,
        load_default=None,
        validate=Range(min=1, min_inclusive=True),
        metadata={
            "label": "Shots",
            "description": "The number of shots to simulate. If execution options are specified they will override this setting!",
            "input_type": "number",
        },
    )
    statevector = ma.fields.Bool(
        required=False,
        allow_none=True,
        load_default=False,
        metadata={
            "label": "Include Statevector",
            "description": "Include a statevector result.",
        },
    )


@BRAKET_LOCAL_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @BRAKET_LOCAL_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @BRAKET_LOCAL_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = Braket_LocalSimulator.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{BRAKET_LOCAL_BLP.name}.ProcessView"),
                ui_href=url_for(f"{BRAKET_LOCAL_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[
                    InputDataMetadata(
                        data_type="executable/circuit",
                        content_type=["text/x-qasm"],
                        required=True,
                        parameter="circuit",
                    ),
                    InputDataMetadata(
                        data_type="provenance/execution-options",
                        content_type=[
                            "text/csv",
                            "application/json",
                            "application/X-lines+json",
                        ],
                        required=False,
                        parameter="executionOptions",
                    ),
                ],
                data_output=[
                    OutputDataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                        name="result-counts.json",
                    ),
                    OutputDataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=False,
                        name="result-statevector.json",
                    ),
                    OutputDataMetadata(
                        data_type="provenance/trace",
                        content_type=["application/json"],
                        required=True,
                        name="result-trace.json",
                    ),
                    OutputDataMetadata(
                        data_type="provenance/execution-options",
                        content_type=["application/json"],
                        required=True,
                        name="execution-options.json",
                    ),
                ],
            ),
            tags=Braket_LocalSimulator.instance.tags,
        )


@BRAKET_LOCAL_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the braket_local simulators plugin."""

    example_inputs: Dict[str, Any] = {
        "shots": 1024,
    }

    @BRAKET_LOCAL_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the braket_local simulators plugin."
    )
    @BRAKET_LOCAL_BLP.arguments(
        BraketSimulatorParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @BRAKET_LOCAL_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        values: ChainMap[str, Any] = ChainMap(request.args.to_dict(), self.example_inputs)
        return self.render(values, errors, False)

    @BRAKET_LOCAL_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the braket_local simulators plugin."
    )
    @BRAKET_LOCAL_BLP.arguments(
        BraketSimulatorParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @BRAKET_LOCAL_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        values: ChainMap[str, Any] = ChainMap(request.form.to_dict(), self.example_inputs)
        return self.render(values, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = Braket_LocalSimulator.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = BraketSimulatorParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{BRAKET_LOCAL_BLP.name}.ProcessView"),
                help_text="",
                example_values=url_for(
                    f"{BRAKET_LOCAL_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@BRAKET_LOCAL_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @BRAKET_LOCAL_BLP.arguments(
        BraketSimulatorParametersSchema(unknown=EXCLUDE), location="form"
    )
    @BRAKET_LOCAL_BLP.response(HTTPStatus.FOUND)
    @BRAKET_LOCAL_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the circuit execution task."""
        db_task = ProcessingTask(
            task_name=execute_circuit.name, parameters=dumps(arguments)
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = execute_circuit.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class Braket_LocalSimulator(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Allows execution of quantum circuits using a simulator packaged with braket_local."
    tags = ["circuit-executor", "qc-simulator", "braket_local", "qasm", "qasm-3"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return BRAKET_LOCAL_BLP

    def get_requirements(self) -> str:
        return "amazon-braket-sdk~=1.66.0\nqiskit~=0.43"


TASK_LOGGER = get_task_logger(__name__)


def postprocess_counts(counts: Dict[str, int], qiskit_circuit) -> Dict[str, int]:
    qubit_positions = {q: i for i, q in enumerate(qiskit_circuit.qubits)}
    measurements = {}

    # TODO: this function misses bits that are not part of registers (only for qasm3)!

    for m in qiskit_circuit.get_instructions("measure"):
        for qb, cb in zip(m.qubits, m.clbits):
            measurements[cb] = qubit_positions[qb]

    def get_bit(count: str, bit) -> str:
        qbit = measurements.get(bit, None)
        if qbit is None:
            return "0"
        return count[qbit]

    def map_count(count: str) -> str:
        return " ".join(
            "".join(get_bit(count, bit) for bit in reversed(reg))
            for reg in reversed(qiskit_circuit.cregs)
        )

    return {map_count(k): v for k, v in counts.items()}


def simulate_circuit(circuit_qasm: str, execution_options: Dict[str, Union[str, int]]):
    from braket.circuits import Instruction, gates
    from braket.circuits import Circuit, ResultType
    from braket.devices import LocalSimulator
    from qiskit import QuantumCircuit
    from qiskit.qasm3 import loads as loads3

    # Convert QASM code to a Cirq circuit
    qiskit_circuit: QuantumCircuit = loads3(circuit_qasm)

    circuit_qasm = circuit_qasm.replace("\r\n", "\n")

    # remove stdgates.inc import to avoid FileNotFoundError

    circuit_qasm = circuit_qasm.replace('include "stdgates.inc";', "")

    # Convert the Cirq circuit to a Braket circuit
    braket_circuit = Circuit.from_ir(circuit_qasm)

    shots = execution_options["shots"]

    metadata = {
        "jobId": "unknown",
        "qpuType": "simulator",
        "qpuVendor": "Amazon Web Services",
        "qpuName": "LocalSimulator",
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "timeTaken": 0,
    }

    if not qiskit_circuit.qubits or not qiskit_circuit.clbits:
        # circuit does not contain any qubit or classical bits
        return metadata, {"": shots}, None
    if not qiskit_circuit.get_instructions("measure"):
        # missing measurement instructions
        return metadata, {"": shots}, None

    for i in range(len(qiskit_circuit.qubits)):
        braket_circuit.add_instruction(Instruction(gates.I(), i))

    device = LocalSimulator()

    # Start the simulation for counts with time nanosecond
    start_time = time.time()
    result_meas = device.run(braket_circuit, shots=execution_options["shots"]).result()
    end_time = time.time()

    result_counts = result_meas.measurement_counts

    result_counts = postprocess_counts(result_counts, qiskit_circuit)

    statevector = None

    # TODO: should be tested for compatibility with other plugins
    if execution_options["statevector"]:
        # Start the simulation for the state vector
        state_vector_circuit = braket_circuit.copy()
        state_vector_circuit.add_result_type(ResultType.StateVector())  # State vector
        state_vector_result = device.run(state_vector_circuit, shots=0).result()
        statevector = [state_vector_result.values[0]]

    metadata["timeTaken"] = end_time - start_time

    return metadata, result_counts, statevector


@CELERY.task(name=f"{Braket_LocalSimulator.instance.identifier}.demo_task", bind=True)
def execute_circuit(self, db_id: int) -> str:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    task_options: Dict[str, Union[str, int]] = loads(task_data.parameters or "{}")
    circuit_url: str = cast(str, task_options["circuit"])

    circuit_qasm: str
    with open_url(circuit_url) as quasm_response:
        circuit_qasm = quasm_response.text

    execution_options_url = cast(
        Optional[str], task_options.get("executionOptions", None)
    )

    execution_options: Dict[str, Any] = {
        "shots": task_options.get("shots", 1),
        "statevector": task_options.get("statevector", False),
    }

    if execution_options_url:
        with open_url(execution_options_url) as execution_options_response:
            try:
                mimetype = execution_options_response.headers["Content-Type"]
            except KeyError:
                mimetype = mimetypes.MimeTypes().guess_type(url=execution_options_url)[0]
            if mimetype is None:
                msg = "Could not guess execution options mime type!"
                TASK_LOGGER.error(msg)
                raise ValueError(msg)  # TODO better error
            entities = ensure_dict(
                load_entities(execution_options_response, mimetype=mimetype)
            )
            options = next(entities, {})
            task_data.add_task_log_entry(
                "loaded execution options: " + dumps(options), commit=True
            )
            execution_options.update(options)

    if isinstance(execution_options["shots"], str):
        execution_options["shots"] = int(execution_options["shots"])

    if isinstance(execution_options["statevector"], str):
        execution_options["statevector"] = execution_options["statevector"].lower() in (
            "1",
            "yes",
            "true",
        )

    metadata, counts, state_vector = simulate_circuit(circuit_qasm, execution_options)

    experiment_id = str(uuid4())

    with SpooledTemporaryFile(mode="w") as output:
        metadata["ID"] = experiment_id
        dump(metadata, output)
        STORE.persist_task_result(
            db_id, output, "result-trace.json", "provenance/trace", "application/json"
        )

    with SpooledTemporaryFile(mode="w") as output:
        counts["ID"] = experiment_id
        dump(counts, output)
        STORE.persist_task_result(
            db_id, output, "result-counts.json", "entity/vector", "application/json"
        )

    if state_vector:
        state_vector_ent = {"ID": experiment_id}
        dim = len(state_vector)
        key_len = len(str(dim))
        for i, v in enumerate(state_vector):
            state_vector_ent[f"{i:0{key_len}}"] = repr(v)
        with SpooledTemporaryFile(mode="w") as output:
            dump(state_vector_ent, output)
            STORE.persist_task_result(
                db_id,
                output,
                "result-statevector.json",
                "entity/vector",
                "application/json",
            )

    extra_execution_options = {
        "ID": experiment_id,
        "executorPlugin": execution_options.get("executorPlugin", []) + [_identifier],
        "shots": metadata.get("shots", execution_options["shots"]),
        "qpuType": metadata["qpuType"],
        "qpuVendor": metadata["qpuVendor"],
        "qpuName": metadata["qpuName"],
        # "qpuVersion": metadata["qpuVersion"],
        "qpuVersion": metadata.get("qpuVersion", "default_value"),
    }

    if "seed" in metadata:
        extra_execution_options["seed"] = metadata["seed"]

    execution_options.update(extra_execution_options)

    with SpooledTemporaryFile(mode="w") as output:
        dump(execution_options, output)
        STORE.persist_task_result(
            db_id,
            output,
            "execution-options.json",
            "provenance/execution-options",
            "application/json",
        )

    return "Finished simulating circuit."
