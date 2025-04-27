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
from os import environ
import time
from collections import ChainMap
from http import HTTPStatus
from json import dump, dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, Mapping, Optional, Union, cast
from uuid import uuid4
from urllib.parse import urljoin
import requests
from requests import RequestException
from time import sleep

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

_plugin_name = "qunicorn-plugin"
__version__ = "v1.0.1"  # ???
_identifier = plugin_identifier(_plugin_name, __version__)

# standard qunicorn options for job submission
QUNICORN_URL = environ.get("QUNICORN_URL", "http://localhost:5001/")
QUNICORN_PROVIDER = environ.get("QUNICORN_PROVIDER", "IBM")
QUNICORN_DEVICE = environ.get("QUNICORN_DEVICE", "aer_simulator")
QUNICORN_TOKEN = environ.get("QUNICORN_TOKEN", "")

QUNICORN_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Circuit executor executing jobs on qunicorn.",
)


class QunicornPluginParametersSchema(FrontendFormBaseSchema):
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
    provider = ma.fields.String(
        required=False,
        allow_none=True,
        load_default=None,
        metadata={
            "label": "Provider",
            "description": "The provider on which the circuit is executed. If execution options are specified they will override this setting",
            "input_type": "text",
        },
    )
    device = ma.fields.String(
        required=False,
        allow_none=True,
        load_default=None,
        metadata={
            "label": "Device",
            "description": "The device on which the circuit is executed. If execution options are specified they will override this setting",
            "input_type": "text",
        },
    )
    token = ma.fields.String(
        required=False,
        allow_none=True,
        load_default=None,
        metadata={
            "label": "Token",
            "description": "The API token for the provider and device specified. If execution options are specified they will override this setting",
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


@QUNICORN_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QUNICORN_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @QUNICORN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = QunicornPlugin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{QUNICORN_BLP.name}.ProcessView"),
                ui_href=url_for(f"{QUNICORN_BLP.name}.MicroFrontend"),
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
            tags=QunicornPlugin.instance.tags,
        )


@QUNICORN_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the qunicorn plugin."""

    example_inputs: Dict[str, Any] = {
        "provider": "IBM",
        "device": "aer_simulator",
        "token": "",
        "shots": 1024,
    }

    @QUNICORN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the qunicorn plugin."
    )
    @QUNICORN_BLP.arguments(
        QunicornPluginParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QUNICORN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        values: ChainMap[str, Any] = ChainMap(request.args.to_dict(), self.example_inputs)
        return self.render(values, errors, False)

    @QUNICORN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the qunicorn plugin."
    )
    @QUNICORN_BLP.arguments(
        QunicornPluginParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QUNICORN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        values: ChainMap[str, Any] = ChainMap(request.form.to_dict(), self.example_inputs)
        return self.render(values, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = QunicornPlugin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = QunicornPluginParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{QUNICORN_BLP.name}.ProcessView"),
                help_text="",
                example_values=url_for(
                    f"{QUNICORN_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@QUNICORN_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @QUNICORN_BLP.arguments(
        QunicornPluginParametersSchema(unknown=EXCLUDE), location="form"
    )
    @QUNICORN_BLP.response(HTTPStatus.FOUND)
    @QUNICORN_BLP.require_jwt("jwt", optional=True)
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


class QunicornPlugin(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Allows execution of quantum circuits using the qunicorn package."
    tags = ["circuit-executor", "qc-simulator", "qunicorn", "qasm", "qasm-2", "qasm-3"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QUNICORN_BLP


TASK_LOGGER = get_task_logger(__name__)


def register_deployment(circuit_qasm: str) -> int:

    is_qasm2 = "OPENQASM 2.0;" in circuit_qasm

    data = {
        "name": "QasmTestsuite Deployment Qhana",
        "programs": [
            {
                "quantumCircuit": circuit_qasm,
                "assemblerLanguage": "QASM2" if is_qasm2 else "QASM3",
            }
        ],
    }
    deployments_url = urljoin(QUNICORN_URL, "/deployments/")
    response = requests.post(deployments_url, json=data)
    response.raise_for_status()
    return response.json()["id"]


def run_job(deployment_id: int, execution_options: Dict[str, Union[str, int]]) -> int:
    data = {
        "name": "QasmTestSuite Job Qhana",
        "providerName": execution_options["provider"],
        "deviceName": execution_options["device"],
        "shots": execution_options["shots"],
        "token": execution_options["token"],
        "type": execution_options["type"],
        "deploymentId": deployment_id,
    }
    deployments_url = urljoin(QUNICORN_URL, "/jobs/")
    response = requests.post(deployments_url, json=data)
    try:
        response.raise_for_status()
    except RequestException as err:
        raise ValueError(response.text) from err
    return response.json()["id"]


def ensure_binary(result: str, counts_format: str, registers: Optional[list[int]]) -> str:
    if counts_format == "bin":
        return result  # result is already binary
    elif counts_format == "hex":
        if registers is None:
            raise ValueError("Parameter registers is required for hex values!")
        register_counts = result.split()
        if len(register_counts) != len(registers):
            raise ValueError(
                "Number of registers in counts string does not match number of registers from metadata!"
            )
        return " ".join(
            f"000{int(val, 16):b}"[-size:]
            for val, size in zip(register_counts, registers)
        )
    return result


def run_qunicorn_circuit(
    circuit_qasm: str, execution_options: Dict[str, Union[str, int]]
) -> Mapping[str, int]:

    deployment_id = register_deployment(circuit_qasm)

    job_id = run_job(deployment_id, execution_options)

    result_url = urljoin(QUNICORN_URL, f"/jobs/{job_id}")

    for i in range(100):
        response = requests.get(result_url, timeout=0.5)
        response.raise_for_status()
        result = response.json()
        if result["state"] == "FINISHED":
            break
        elif result["state"] in ("ERROR", "CANCELED"):
            raise ValueError(f"Qunicorn job ended with a Failure! ({result_url})")
        sleep(1)
    if result["state"] == "FINISHED":
        counts = None
        probabilities = None
        no_count = None
        no_prob = None
        registers = None
        counts_format = "bin"
        for output in result["results"]:
            if output["resultType"] == "COUNTS":
                counts = output["data"]
                metadata = output["metadata"]
                if metadata.get("format") in ("bin", "hex"):
                    counts_format = metadata["format"]
                if counts_format == "hex":
                    registers = [r["size"] for r in metadata["registers"]]
            if output["resultType"] == "PROBABILITIES":
                probabilities = output["data"]
                metadata_prob = output["metadata"]
                if metadata_prob.get("format") in ("bin", "hex"):
                    prob_format = metadata_prob["format"]
                if counts_format == "hex":
                    registers = [r["size"] for r in metadata_prob["registers"]]
        if counts:
            counts = {
                ensure_binary(k, counts_format, registers): v for k, v in counts.items()
            }
            if set(counts.keys()) == {""}:
                no_count = True
                counts_keys = {}

        if probabilities:
            probabilities = {
                ensure_binary(k, prob_format, registers): v
                for k, v in probabilities.items()
            }
            if set(probabilities.keys()) == {""}:
                no_prob = True
                prob_keys = {}

        if no_count and no_prob:
            return counts_keys, prob_keys
        elif counts and probabilities:
            return counts, probabilities
        else:
            raise ValueError(f"Did not produce any counts! ({result_url})")

    elif result["state"] in ("ERROR", "CANCELED"):
        raise ValueError(f"Qunicorn job ended with a Failure! ({result_url})")
    else:
        raise ValueError(f"Qunicorn job timed out producing a result! ({result_url})")


def bin_to_hex(binary_str: str):
    decimal = int(binary_str, 2)
    hex_value = hex(decimal)[2:]
    hex_value = hex_value.upper()
    return hex_value


def state_from_prob(probabilities):
    """
    calculate statevector from probabilities as qunicorn only returns counts and probabilities
    """
    import numpy as np

    binary_key = list(probabilities.keys())[0]
    len_statevector = 2 ** len(binary_key)
    statevector = np.zeros(len_statevector, dtype=complex)
    for key in probabilities.keys():
        i = int(bin_to_hex(key))
        value = probabilities[key]
        statevector[i] = np.sqrt(value)
    return statevector


def simulate_circuit(circuit_qasm: str, execution_options: Dict[str, Union[str, int]]):

    shots = execution_options["shots"]
    provider = execution_options["provider"]
    device = execution_options["device"]
    token = execution_options["token"]

    metadata = {
        "qpuType": "simulator",
        "qpuVendor": "Xanadu Inc",
        "provider": provider,
        "device": device,
        "token": token,
        "shots": shots,
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "timeTakenCounts_nanosecond": 0,
    }

    startime_counts = time.perf_counter_ns()
    counts, probabilities = run_qunicorn_circuit(circuit_qasm, execution_options)
    endtime_counts = time.perf_counter_ns()
    metadata["timeTakenCounts_nanosecond"] = endtime_counts - startime_counts

    if execution_options.get("statevector"):
        result_state = state_from_prob(probabilities)
    else:
        result_state = None

    return metadata, counts, result_state


@CELERY.task(name=f"{QunicornPlugin.instance.identifier}.demo_task", bind=True)
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
        "provider": task_options.get("provider"),
        "device": task_options.get("device"),
        "token": task_options.get("token"),
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
    if "provider" not in execution_options:
        execution_options["provider"] = QUNICORN_PROVIDER
    if "device" not in execution_options:
        execution_options["device"] = QUNICORN_DEVICE
    if "token" not in execution_options:
        execution_options["token"] = QUNICORN_TOKEN
    if "type" not in execution_options:
        execution_options["type"] = "RUNNER"
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

    if counts:
        counts_ent = {key: int(value) for key, value in counts.items()}
        counts_ent["ID"] = experiment_id

        with SpooledTemporaryFile(mode="w") as output:
            dump(counts_ent, output)
            STORE.persist_task_result(
                db_id,
                output,
                "result-counts.json",
                "entity/vector",
                "application/json",
            )
    else:
        raise ValueError("Failed to simulate circuit. No counts are available.")

    if state_vector is not None and any(state_vector):
        str_vector = [str(x) for x in state_vector.tolist()]

        state_vector_ent = {"ID": experiment_id}
        dim = len(str_vector)
        key_len = len(str(dim))
        for i, v in enumerate(str_vector):
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


# try:
#     # import for type annotations
#     from qiskit import QuantumCircuit
# except ImportError:
#     pass
