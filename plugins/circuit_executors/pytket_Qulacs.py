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
from typing import Any, Dict, Mapping, Optional, Tuple, Union, cast
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

_plugin_name = "pytket_qulacsBackend-simulator"
__version__ = "v1.0.0"
_identifier = plugin_identifier(_plugin_name, __version__)


PYTKET_QULACSBACKEND_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Circuit executor exposing the pytket_qulacsBackend simulators as backend.",
)


class Pytket_qulacsBackendSimulatorParametersSchema(FrontendFormBaseSchema):
    circuit = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="executable/circuit",
        data_content_types="text/x-qasm",
        metadata={
            "label": "OpenQASM Circuit",
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


@PYTKET_QULACSBACKEND_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PYTKET_QULACSBACKEND_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @PYTKET_QULACSBACKEND_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = Pytket_qulacsBackendSimulator.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{PYTKET_QULACSBACKEND_BLP.name}.ProcessView"),
                ui_href=url_for(f"{PYTKET_QULACSBACKEND_BLP.name}.MicroFrontend"),
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
            tags=Pytket_qulacsBackendSimulator.instance.tags,
        )


@PYTKET_QULACSBACKEND_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the pytket_qulacsBackend simulators plugin."""

    example_inputs: Dict[str, Any] = {
        "shots": 1024,
    }

    @PYTKET_QULACSBACKEND_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the pytket_qulacsBackend simulators plugin.",
    )
    @PYTKET_QULACSBACKEND_BLP.arguments(
        Pytket_qulacsBackendSimulatorParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @PYTKET_QULACSBACKEND_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        values: ChainMap[str, Any] = ChainMap(request.args.to_dict(), self.example_inputs)
        return self.render(values, errors, False)

    @PYTKET_QULACSBACKEND_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the pytket_qulacsBackend simulators plugin.",
    )
    @PYTKET_QULACSBACKEND_BLP.arguments(
        Pytket_qulacsBackendSimulatorParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @PYTKET_QULACSBACKEND_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        values: ChainMap[str, Any] = ChainMap(request.form.to_dict(), self.example_inputs)
        return self.render(values, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = Pytket_qulacsBackendSimulator.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = Pytket_qulacsBackendSimulatorParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{PYTKET_QULACSBACKEND_BLP.name}.ProcessView"),
                help_text="",
                example_values=url_for(
                    f"{PYTKET_QULACSBACKEND_BLP.name}.MicroFrontend",
                    **self.example_inputs,
                ),
            )
        )


@PYTKET_QULACSBACKEND_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @PYTKET_QULACSBACKEND_BLP.arguments(
        Pytket_qulacsBackendSimulatorParametersSchema(unknown=EXCLUDE), location="form"
    )
    @PYTKET_QULACSBACKEND_BLP.response(HTTPStatus.FOUND)
    @PYTKET_QULACSBACKEND_BLP.require_jwt("jwt", optional=True)
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


class Pytket_qulacsBackendSimulator(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Allows execution of quantum circuits using a simulator packaged with pytket_qulacsBackend."
    tags = ["circuit-executor", "qc-simulator", "pytket_qulacsBackend", "qasm", "qasm-2"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return PYTKET_QULACSBACKEND_BLP

    def get_requirements(self) -> str:
        return "pytket-qulacs~=0.29.0\nqiskit_qasm3_import\npytket<1.36"


TASK_LOGGER = get_task_logger(__name__)


def postprocess_counts(
    counts: Dict[Tuple[int, ...], int], qubits_readout, c_registers, mapping
) -> Dict[str, int]:
    """Map the tuples in the count dictionary to qiskit style string results.

    Args:
        counts (Dict[Tuple[int, ...], int]): the counts dictionary containing tuples of qbit measurements
        qubits_readout (Dict[qubit, int]): the dictionary how the qubits are read out from the circuit
        c_registers (List[BitRegister]): the classical registers in the order they appear in the result string
        mapping (Dict[qubit, bit]): the mapping from qubits to classical bits

    Returns:
        Dict[str, int]: the qiskit style counts dictionary
    """
    reversed_mapping = {}
    for qb, b in mapping.items():
        reversed_mapping[b] = qb

    # qubits that are not measured are not included in the result tuple, even if they are
    # included in the qubit readout mapping!
    # -> remove qubits that were not measured (by checking the mapping)
    # -> keep the order of qubits from the readout mapping
    used_qubits = sorted(
        [q for q in qubits_readout if q in mapping], key=lambda q: qubits_readout[q]
    )
    qbit_to_int = {q: i for i, q in enumerate(used_qubits)}

    def get_result(count: Tuple[int, ...], bit) -> str:
        qbit = reversed_mapping.get(bit, None)
        if qbit is None:
            return "0"
        return str(count[qbit_to_int[qbit]])

    def map_result(count: Tuple[int, ...]):
        return " ".join(
            "".join(get_result(count, b) for b in reversed(reg)) for reg in c_registers
        )

    return {map_result(v): c for v, c in counts.items()}


def simulate_circuit(circuit_qasm: str, execution_options: Dict[str, Union[str, int]]):
    from pytket.circuit import Circuit
    from pytket.extensions.qulacs import QulacsBackend
    from pytket.qasm import circuit_from_qasm_str

    shots = execution_options["shots"]

    metadata = {
        "qpuType": "simulator",
        "qpuVendor": "Quantinuum & Qulacs Team",
        "qpuName": "QulacsBackend",
        "qpuVersion": None,
        "shots": "shots",
        "timeTaken": 0,
        "timeTakenIdle": 0,
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    startime = time.time()
    # Convert circuit from qasm code
    circ: Circuit = circuit_from_qasm_str(circuit_qasm)

    def c_reg_pos(c_reg) -> int:
        """Find the position of the classical register definition in the qasm string."""
        # TODO: the regex may find definitions that were commented out...
        match = re.search(f"creg\\s+{c_reg.name}\\[{c_reg.size}\\]\\s*;", circuit_qasm)
        if match:
            return match.start()
        return len(circuit_qasm)

    # sort classical registers by appeareance in the qasm definition
    c_registers = sorted(circ.c_registers, key=c_reg_pos, reverse=True)

    if not circ.qubit_to_bit_map:
        return metadata, {"": shots}, None

    backend = QulacsBackend()
    # compiled circuit to be ready simulations with Backend
    compiled_circuit = backend.get_compiled_circuit(circ)

    startime_counts = time.perf_counter_ns()
    handle = backend.process_circuit(
        compiled_circuit, n_shots=execution_options["shots"]
    )  # count simulation with time
    result = backend.get_result(handle)
    endtime_counts = time.perf_counter_ns()

    # get result and transform counts to qiskit style counts
    counts = dict(result.get_counts())
    counts = postprocess_counts(
        counts, circ.qubit_readout, c_registers, circ.qubit_to_bit_map
    )

    if execution_options.get("statevector"):
        # only execute if statevector result was requested in the first place
        statevector = backend.get_result(handle).get_state()
    else:
        statevector = None

    endtime = time.time()

    simulation_time = endtime - startime

    metadata.update(
        {
            "timeTaken": simulation_time,
            "timeTakenQpu": simulation_time,
            "timeTaken_Counts_nanosecond": endtime_counts - startime_counts,
        }
    )

    return metadata, counts, statevector


@CELERY.task(
    name=f"{Pytket_qulacsBackendSimulator.instance.identifier}.demo_task", bind=True
)
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
        "statevector": bool(task_options.get("statevector")),
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
        execution_options["statevector"] = execution_options["ststevector"] in (
            "1",
            "yes",
            "Yes",
            "YES",
            "true",
            "True",
            "TRUE",
        )

    metadata, counts, state_vector = simulate_circuit(circuit_qasm, execution_options)

    experiment_id = str(uuid4())

    with SpooledTemporaryFile(mode="w") as output:
        metadata["ID"] = experiment_id
        dump(metadata, output)
        STORE.persist_task_result(
            db_id, output, "result-trace.json", "provenance/trace", "application/json"
        )

    # FIXME check if bit order is consistent withother simulators!!!
    counts_str_keys = {
        "".join(str(b) for b in key): int(value) for key, value in counts.items()
    }

    with SpooledTemporaryFile(mode="w") as output:
        counts_str_keys["ID"] = experiment_id
        dump(counts_str_keys, output)
        STORE.persist_task_result(
            db_id, output, "result-counts.json", "entity/vector", "application/json"
        )

    if state_vector is not None and state_vector.any():
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
        "qpuVersion": metadata["qpuVersion"],
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
