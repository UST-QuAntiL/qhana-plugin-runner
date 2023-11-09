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


import mimetypes
from http import HTTPStatus
from json import dump, dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Any, ChainMap, Dict, Mapping, Optional, Union, cast
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

_plugin_name = "cirq-simulator"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

CIRQ_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Circuit executor exposing the cirq simulators as backend.",
)


class CirqSimulatorParametersSchema(FrontendFormBaseSchema):
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


@CIRQ_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @CIRQ_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @CIRQ_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = CirqSimulator.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{CIRQ_BLP.name}.ProcessView"),
                ui_href=url_for(f"{CIRQ_BLP.name}.MicroFrontend"),
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
            tags=CirqSimulator.instance.tags,
        )


################


@CIRQ_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the cirq simulators plugin."""

    example_inputs: Dict[str, Any] = {
        "shots": 1024,
    }

    @CIRQ_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the cirq simulators plugin."
    )
    @CIRQ_BLP.arguments(
        CirqSimulatorParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @CIRQ_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        values: ChainMap[str, Any] = ChainMap(request.args.to_dict(), self.example_inputs)
        return self.render(values, errors, False)

    @CIRQ_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the cirq simulators plugin."
    )
    @CIRQ_BLP.arguments(
        CirqSimulatorParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @CIRQ_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        values: ChainMap[str, Any] = ChainMap(request.form.to_dict(), self.example_inputs)
        return self.render(values, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = CirqSimulator.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = CirqSimulatorParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{CIRQ_BLP.name}.ProcessView"),
                help_text="",
                example_values=url_for(
                    f"{CIRQ_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@CIRQ_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @CIRQ_BLP.arguments(CirqSimulatorParametersSchema(unknown=EXCLUDE), location="form")
    @CIRQ_BLP.response(HTTPStatus.FOUND)
    @CIRQ_BLP.require_jwt("jwt", optional=True)
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


class CirqSimulator(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "Allows execution of quantum circuits using a simulator packaged with cirq."
    )
    tags = ["circuit-executor", "qc-simulator", "cirq", "qasm", "qasm-2"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return CIRQ_BLP

    def get_requirements(self) -> str:
        return "cirq~=1.3.0.dev20230804185427"


TASK_LOGGER = get_task_logger(__name__)


# regex to find total number of classicalbits
def find_total_classicalbits(qasm_code):
    import re

    cleancomment_qasm = re.sub(r"//.*\n?", "", qasm_code)
    matches = re.findall(r"creg [a-zA-Z0-9_]+\[(\d+)\];", cleancomment_qasm)
    if not matches:
        return 0
    return sum(map(int, matches))


def simulate_circuit(circuit_qasm: str, execution_options: Dict[str, Union[str, int]]):
    from cirq import Circuit, Simulator
    from cirq.contrib.qasm_import import circuit_from_qasm
    import time
    import cirq
    import numpy as np

    circuit_qasm = circuit_qasm.replace("\r\n", "\n")
    circuit = circuit_from_qasm(circuit_qasm)
    # Zero time indicates no measurements (in qasm code)
    startime_count = 0
    endtime_count = 0
    # Make a copy of the circuit to keep original, unchanged before adding any measuremnts.
    circuit_copy = circuit.copy()

    number_qubits = len(list(circuit.all_qubits()))

    num_classicalbits = find_total_classicalbits(circuit_qasm)
    has_measurem = any(
        isinstance(op.gate, cirq.MeasurementGate) for op in circuit.all_operations()
    )  # Check if the circuit includes any measurement Gates

    simulator = cirq.Simulator()

    startime_state = time.perf_counter_ns()
    state_vector = simulator.simulate(
        circuit
    ).final_state_vector  # simulation (statevector)
    endtime_state = time.perf_counter_ns()

    histogram = {}
    if (
        has_measurem or num_classicalbits > 0
    ):  # If the circuit (qasm code) has measurements  measure all qubits
        circuit_copy.append(cirq.measure(*circuit.all_qubits(), key="result"))

        startime_count = time.perf_counter_ns()
        result_count = simulator.run(
            circuit_copy, repetitions=execution_options["shots"]
        )  # simulation (counts)
        endtime_count = time.perf_counter_ns()

        histogram = result_count.histogram(key="result")
    else:  # If there are no measurements in qasm code
        shots = execution_options["shots"]
        histogram = {"": shots}

    # Convert the outcomes to binary format
    binary_histogram = {
        format(outcome, f"0{number_qubits}b")
        if outcome and isinstance(outcome, int)
        else outcome: frequency
        for outcome, frequency in histogram.items()
    }

    metadata = {
        # trace ids (specific to IBM qiskit jobs)
        "jobId": None,
        "qobjId": None,
        # QPU/Simulator information
        "qpuType": "simulator",
        "qpuVendor": "Google",
        "qpuName": "Cirq Simulator",
        "qpuVersion": None,
        "seed": None,  # only for simulators
        "shots": execution_options["shots"],
        # Time information
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "timeTakenIdle": 0,  # idle/waiting time
        "timeTakenCounts_nanosecond": endtime_count - startime_count,
        "timeTakenState_nanosecond": endtime_state - startime_state,
    }

    return metadata, binary_histogram, state_vector


@CELERY.task(name=f"{CirqSimulator.instance.identifier}.demo_task", bind=True)
def execute_circuit(self, db_id: int) -> str:
    import numpy as np

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
                raise ValueError(msg)
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

    metadata, binary_histogram, state_vector = simulate_circuit(
        circuit_qasm, execution_options
    )

    binary_histogram = {k: int(v) for k, v in binary_histogram.items()}

    experiment_id = str(uuid4())

    with SpooledTemporaryFile(mode="w") as output:
        metadata["ID"] = experiment_id
        dump(metadata, output)
        STORE.persist_task_result(
            db_id, output, "result-trace.json", "provenance/trace", "application/json"
        )

    with SpooledTemporaryFile(mode="w") as output:
        binary_histogram["ID"] = experiment_id
        dump(binary_histogram, output)
        STORE.persist_task_result(
            db_id, output, "result-counts.json", "entity/vector", "application/json"
        )

    # Finally, #if the conditions are ok, the `state_vector` is converted into a list,  and each element is transformed into a string, This can be used to convert complex numbers into the string format.

    if state_vector is not None and np.any(state_vector):
        str_vector = [str(x) for x in state_vector.tolist()]  #### tolist here ok!
        state_vector_ent = {"ID": experiment_id, "statevector": str_vector}

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
        "simulator": metadata["qpuType"],
        "qpuType": metadata["qpuType"],
        "qpuVendor": metadata["qpuVendor"],
        "qpuName": metadata["qpuName"],
        "qpuVersion": metadata["qpuVersion"],
    }

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
