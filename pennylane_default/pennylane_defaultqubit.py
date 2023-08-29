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

_plugin_name = "pennylane-simulator"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


PENNYLANE_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Circuit executor exposing the qiskit simulators as backend.",
)


class PennylaneSimulatorParametersSchema(FrontendFormBaseSchema):
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


@PENNYLANE_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PENNYLANE_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @PENNYLANE_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = PennylaneSimulator.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{PENNYLANE_BLP.name}.ProcessView"),
                ui_href=url_for(f"{PENNYLANE_BLP.name}.MicroFrontend"),
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
            tags=PennylaneSimulator.instance.tags,
        )


@PENNYLANE_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the qiskit simulators plugin."""

    example_inputs: Dict[str, Any] = {
        "shots": 1024,
    }

    @PENNYLANE_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the qiskit simulators plugin."
    )
    @PENNYLANE_BLP.arguments(
        PennylaneSimulatorParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @PENNYLANE_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        values: ChainMap[str, Any] = ChainMap(request.args.to_dict(), self.example_inputs)
        return self.render(values, errors, False)

    @PENNYLANE_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the pennylane simulators plugin."
    )
    @PENNYLANE_BLP.arguments(
        PennylaneSimulatorParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @PENNYLANE_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        values: ChainMap[str, Any] = ChainMap(request.form.to_dict(), self.example_inputs)
        return self.render(values, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = PennylaneSimulator.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = PennylaneSimulatorParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{PENNYLANE_BLP.name}.ProcessView"),
                help_text="",
                example_values=url_for(
                    f"{PENNYLANE_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@PENNYLANE_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @PENNYLANE_BLP.arguments(
        PennylaneSimulatorParametersSchema(unknown=EXCLUDE), location="form"
    )
    @PENNYLANE_BLP.response(HTTPStatus.FOUND)
    @PENNYLANE_BLP.require_jwt("jwt", optional=True)
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


class PennylaneSimulator(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "Allows execution of quantum circuits using a simulator packaged with qiskit."
    )
    tags = ["circuit-executor", "qc-simulator", "pennylane", "qasm", "qasm-2"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return PENNYLANE_BLP

    def get_requirements(self) -> str:

        return """pennylane~=0.31.0"""


TASK_LOGGER = get_task_logger(__name__)


def extract_total_qubits(qasm_code):
    # regex
    import re

    cleaned_qasm = re.sub(r"//.*\n?", "", qasm_code)

    matches = re.findall(r"qreg [a-zA-Z0-9_]+\[(\d+)\];", cleaned_qasm)

    return sum(map(int, matches))


def simulate_circuit(circuit_qasm: str, execution_options: Dict[str, Union[str, int]]):

    import pennylane as qml
    import time

    ### num_wires = int(circuit_qasm.split('qreg q[')[1].split(']')[0])

    num_wires = extract_total_qubits(circuit_qasm)

    circ = qml.device("default.qubit", wires=num_wires, shots=execution_options["shots"])
    circ_statevector = qml.device("default.qubit", wires=num_wires, shots=None)

    my_circuit = qml.from_qasm(circuit_qasm)

    @qml.qnode(circ)
    def circuit():
        my_circuit(wires=range(num_wires))
        return qml.counts()

    @qml.qnode(circ)
    def state_vector_circuit():
        my_circuit(wires=range(num_wires))
        return qml.state()

    start_time = time.time()
    result = circuit()
    end_time = time.time()

    simulation_time = end_time - start_time

    metadata = {
        "qpuType": "simulator",
        "qpuVendor": "Xanadu Inc",
        "shots": execution_options["shots"],
        "timeTaken": simulation_time,
    }

    counts = dict(result)

    # state_vector: Optional[Any] = None

    state_vector = state_vector_circuit()

    return metadata, dict(counts), state_vector


@CELERY.task(name=f"{PennylaneSimulator.instance.identifier}.demo_task", bind=True)
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

    metadata, counts, state_vector = simulate_circuit(circuit_qasm, execution_options)

    experiment_id = str(uuid4())

    with SpooledTemporaryFile(mode="w") as output:
        metadata["ID"] = experiment_id
        dump(metadata, output)
        STORE.persist_task_result(
            db_id, output, "result-trace.json", "provenance/trace", "application/json"
        )

    if counts is not None and counts:
        counts_t_strg = {key: str(value) for key, value in counts.items()}
        counts_result = {"ID": experiment_id, "counts": counts_t_strg}

        with SpooledTemporaryFile(mode="w") as output:
            dump(counts_result, output)
            STORE.persist_task_result(
                db_id,
                output,
                "result-counts.json",
                "entity/vector",
                "application/json",
            )

    # with SpooledTemporaryFile(mode="w") as output:
    #   counts["ID"] = experiment_id
    #  dump(counts, output)
    # STORE.persist_task_result(
    #    db_id, output, "result-counts.json", "entity/vector", "application/json"
    # )

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
        "qpuType": metadata["qpuType"],
        "qpuVendor": metadata["qpuVendor"],
        "qpuName": metadata.get("qpuName", "default_value"),
        "qpuVersion": metadata.get("qpuVersion", "default_version_value"),
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
