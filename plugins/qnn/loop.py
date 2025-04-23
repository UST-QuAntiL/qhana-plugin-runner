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
from textwrap import dedent
from json import loads
from tempfile import SpooledTemporaryFile
from qhana_plugin_runner import db
from qhana_plugin_runner.storage import STORE
from typing import Mapping, Optional, cast, Dict, Union
import json

# Adding optimizer to loop
from enum import Enum
from qhana_plugin_runner.api.extra_fields import EnumField, CSVList
from scipy.optimize import minimize
import numpy.typing as npt
from collections.abc import Callable
from requests import get
from time import sleep

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.qasm3 import dumps
from qiskit_qasm3_import import parse

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import abort, jsonify, redirect
from flask.app import Flask
from flask.globals import current_app, request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE, fields

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    MaBaseSchema,
    InputDataMetadata,
    OutputDataMetadata,
    PluginDependencyMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.plugin_utils.interop import (
    call_plugin_endpoint,
    get_plugin_endpoint,
    get_task_result_no_wait,
    monitor_external_substep,
    monitor_result,
    subscribe,
)

from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    PluginUrl,
    FileUrl,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.tasks import (
    TASK_DETAILS_CHANGED,
    TASK_STEPS_CHANGED,
    save_task_error,
    save_task_result,
)

_plugin_name = "loop"
__version__ = "v0.1"
_identifier = plugin_identifier(_plugin_name, __version__)


LOOP_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Loop plugin to orchestrate.",
    template_folder="loop_templates",
)


class OPTIMIZERENUM(Enum):
    SPSA = "SPSA (Simultaneous Perturbation Stochastic Approximation)"
    COBYLA = "COBYLA (Constrained Optimization BY Linear Approximation)"


class LoopParametersSchema(FrontendFormBaseSchema):
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

    executor = PluginUrl(
        required=True,
        plugin_tags=["circuit-executor", "qasm-3"],
        metadata={
            "label": "Select Circuit Executor Plugin",
        },
    )

    state = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="executable/circuit",
        data_content_types="text/x-qasm",
        metadata={
            "label": "State",
            "description": "URL to a quantum circuit in the OpenQASM format.",
            "input_type": "text",
        },
    )

    ansatz = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="executable/circuit",
        data_content_types="text/x-qasm",
        metadata={
            "label": "Ansatz",
            "description": "URL to a quantum circuit in the OpenQASM format.",
            "input_type": "text",
        },
    )

    statevector = fields.Bool(
        required=False,
        missing=False,
        metadata={
            "label": "Request statevector",
        },
    )

    target_statevector = fields.String(
        required=False,
        allow_none=True,
        element_type=fields.String,
        metadata={
            "label": "Target Statevector",
            "description": "State vector that the ansatz should produce. Differen values expected as list of list [[real, imag], [real, imag], ...]",
            "input_type": "textarea",
        },
    )


class WebhookParams(MaBaseSchema):
    source = fields.URL()
    event = fields.String()


class ExecutionOptionsParams(MaBaseSchema):
    statevector = fields.Bool(missing=False)


@LOOP_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @LOOP_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @LOOP_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = Loop.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{LOOP_BLP.name}.ProcessView"),
                ui_href=url_for(f"{LOOP_BLP.name}.MicroFrontend"),
                plugin_dependencies=[
                    PluginDependencyMetadata(
                        required=True,
                        parameter="executor",
                        tags=["circuit-executor", "qasm-3"],
                    ),
                ],
                # TODO add optimizer and target state vector
                data_input=[
                    InputDataMetadata(
                        data_type="executable/circuit",
                        content_type=["text/x-qasm"],
                        required=True,
                        parameter="state",
                    ),
                    InputDataMetadata(
                        data_type="executable/circuit",
                        content_type=["text/x-qasm"],
                        required=True,
                        parameter="ansatz",
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
                        data_type="executable/circuit",
                        content_type=["text/x-qasm"],
                        required=True,
                        name="circuit.qasm",
                    ),
                ],
            ),
            tags=Loop.instance.tags,
        )


@LOOP_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the state preparation plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @LOOP_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the loop plugin."
    )
    @LOOP_BLP.arguments(
        LoopParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @LOOP_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @LOOP_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the loop plugin."
    )
    @LOOP_BLP.arguments(
        LoopParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @LOOP_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = Loop.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = LoopParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{LOOP_BLP.name}.ProcessView"),
                help_text="",
                example_values=url_for(
                    f"{LOOP_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@LOOP_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @LOOP_BLP.arguments(LoopParametersSchema(unknown=EXCLUDE), location="form")
    @LOOP_BLP.response(HTTPStatus.SEE_OTHER)
    @LOOP_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the loop task."""

        # FIXME for some reason optimize_ansatz.name cannot be found
        db_task = ProcessingTask(
            task_name=optimize_ansatz.name,
            parameters=LoopParametersSchema().dumps(arguments),
        )
        # TODO check if both are actually required
        db_task.save()
        DB.session.flush()

        print(OPTIMIZERENUM.COBYLA)

        statevector: bool = arguments.get("statevector", False)

        options_url = url_for(
            f"{LOOP_BLP.name}.{ExecutionOptionsView.__name__}",
            statevector=statevector,
            _external=True,
        )

        db_task.data = {"circuit_string": "circuit goes here"}

        circuit_url = url_for(
            f"{LOOP_BLP.name}.{PrepareCircuitView.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        db_task.data["options_url"] = options_url
        db_task.data["circuit_url"] = circuit_url

        print("circuit_url", circuit_url)

        db_task.save(commit=True)

        # NOTE Execution of ProcessingTask starts here (?)
        # all tasks need to know about db id to load the db entry
        task = optimize_ansatz.s(db_id=db_task.id)

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@LOOP_BLP.route("/continue/<int:db_id>/")
class ContinueProcessView(MethodView):
    """Restart long running task that was blocked by an ongoing plugin computation."""

    @LOOP_BLP.arguments(WebhookParams(partial=True), location="query")
    @LOOP_BLP.response(HTTPStatus.NO_CONTENT)
    def post(self, params: dict, db_id: int):
        """Check for updates in plugin computation and resume processing."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if task_data is None:
            abort(HTTPStatus.NOT_FOUND)

        if task_data.task_name != optimize_ansatz.name:
            # processing task is from another plugin, cannot resume
            abort(HTTPStatus.NOT_FOUND)

        if not isinstance(task_data.data, dict):
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        event_source = params.get("source", None)
        event_type = params.get("event", None)

        result_url = task_data.data.get("result_url")

        if event_source != result_url:
            abort(HTTPStatus.NOT_FOUND)

        if not result_url or task_data.is_finished:
            abort(HTTPStatus.NOT_FOUND)

        task = check_executor_result_task.s(db_id=db_id, event_type=event_type)
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


@LOOP_BLP.route("/options/")
class ExecutionOptionsView(MethodView):
    """Get the execution options."""

    @LOOP_BLP.arguments(ExecutionOptionsParams(), location="query", as_kwargs=True)
    def get(self, statevector: bool = False):
        """Get the requested execution options."""

        return jsonify({"ID": "1", "shots": 2048, "statevector": statevector})


class Loop(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Orchestrates the optimization Loop."
    tags = ["Loop", "Orchestrattion"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return LOOP_BLP

    # scipy einbinden
    def get_requirements(self) -> str:
        return "qiskit~=1.3.2\nnumpy"


@LOOP_BLP.route("/circuit/<int:db_id>")
class PrepareCircuitView(MethodView):
    """Get the circuit as string in a Response."""

    # NOTE maybe db_id should be str
    def get(self, db_id: int) -> Response:
        """Get the requested circuit."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

        return Response(
            task_data.data["circuit_string"],
            HTTPStatus.OK,
            mimetype="text/x-qasm",
        )


# NOTE maybe a new type statevector would be better
def get_cost_function(
    # ansatz: qasam_url, ansatz_plugin_url??  # TODO decide
    db_id: int,
    circuit: QuantumCircuit,
    # target_statevector: npt.NDArray[np.complex128],
) -> Callable[[npt.NDArray[np.float64]], float]:
    """
    Returns the cost/loss function taking single argument, i.e. (ansatz) parameters, which will be
    minimized during optimization.
    """
    # TODO find out how to log/where logs are
    TASK_LOGGER.info("Get cost function successfully called.")
    print("Print: Get cost function successfully called.")

    # NOTE maybe not accessing task_data here and instead passing everything would be nicer
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    task_options: Dict[str, Union[str, int]] = LoopParametersSchema().loads(
        task_data.parameters or "{}"
    )
    executor: Optional[str] = cast(str, task_options["executor"])

    # I think this check is unnecessary as executor field can't be empty?
    # Otherwise I feel like this should be checked generally at a dedicated place
    # - Lukas
    if executor is None:
        task_data.add_task_log_entry(
            "No executor plugin specified, aborting task.", commit=True
        )
        raise ValueError(
            "Cannot execute a quantum circuit without a circuit executor plugin specified."
        )

    endpoint = get_plugin_endpoint(executor)
    options_url = task_data.data["options_url"]
    print("Options URL", options_url)

    target_statevector = np.array(
        [complex(real, imag) for real, imag in eval(task_options["target_statevector"])],
        dtype=np.complex128,
    )

    TASK_LOGGER.debug("target_statevector", target_statevector)

    def cost_function(params: npt.NDArray[np.float64]) -> float:
        # later iterate over list of inputs

        TASK_LOGGER.info("Cost function successfully called.")
        print("COST function successfully called.")
        parametrized_circuit = circuit.assign_parameters(params)

        task_data.data["circuit_string"] = circuit_to_qasm3_string(parametrized_circuit)
        task_data.save(commit=True)

        circuit_url = task_data.data["circuit_url"]

        print("Succesfully built circuit_url", circuit_url)

        result_url = call_plugin_endpoint(
            endpoint, {"circuit": circuit_url, "executionOptions": options_url}
        )

        pending = True

        # TODO make this more pretty and robust - see check_executor_result_task()
        # count = 0
        while pending:
            sleep(0.05)  # NOTE might not be ideal depening on system
            result = loads(get(result_url).text)
            if result["status"] == "SUCCESS":
                stv_url = result["outputs"][2]["href"]
                # print("statevector url:", stv_url)
                statevector_dict = loads(get(stv_url).text)

                # TODO figure out how to access logstream
                # TASK_LOGGER.info("statevector_dict:", statevector_dict)

                _ = statevector_dict.pop("ID")
                # print("statevector_id", statevector_id)

                # this check is probably unneccesary
                if len(statevector_dict) % 2 != 0:
                    raise ValueError(
                        "The returned statevector is missing an imaginary number (legnth is uneven)."
                    )
                elif len(statevector_dict) != len(target_statevector):
                    raise ValueError(
                        "Size of target statevector does not match the size of ansatz statevector."
                    )

                statevector = np.empty(len(statevector_dict), dtype=np.complex128)
                for i in range(len(statevector_dict)):
                    statevector[i] = complex(statevector_dict[str(i)])

                # print("statevector:", statevector)

                # TODO get statevector url call it and get statevector...
                pending = False
            elif result["status"] == "FAILURE":
                # TODO handle this, i.e make sure it shows properly in UI
                raise Exception(f"Executing circuit failed with response: {result}")

            # count +=1

        # print("nr. of waiting loops:", count)

        # Fidelity = |<result_statevector|target_statevector>|^2
        fidelity = np.abs(np.matrix(statevector) @ np.matrix(target_statevector).T) ** 2

        return 1 - fidelity

    return cost_function


def combine_circuit(db_id: int) -> QuantumCircuit:
    """
    This function gets the ID of an optimze_ansatz ProcessingTask and combines Viewthe in the task.data
    linked qasm code for encoding the input data and the actual ansatz. It then returns the
    combined qasm circuit.
    """
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    params = LoopParametersSchema().loads(task_data.parameters or "{}")
    state_url: Optional[str] = params.get("state", None)
    ansatz_url: Optional[str] = params.get("ansatz", None)

    ansatz_qasm: str
    with open_url(ansatz_url) as quasm_response:
        ansatz_qasm = quasm_response.text
        ansatz_circuit = parse(ansatz_qasm)

    state_qasm: str
    with open_url(state_url) as quasm_response:
        state_qasm = quasm_response.text
        state_circuit = parse(state_qasm)

    if state_qasm is None or ansatz_qasm is None:
        raise ValueError("Cannot execute a quantum circuit without a circuit specified.")

    total_qubits = max(ansatz_circuit.num_qubits, state_circuit.num_qubits)
    total_clbits = max(ansatz_circuit.num_clbits, state_circuit.num_clbits)
    combined_circuit = QuantumCircuit(total_qubits, total_clbits)
    combined_circuit.compose(state_circuit, inplace=True)
    combined_circuit.compose(
        ansatz_circuit,
        qubits=range(total_qubits),
        clbits=range(total_clbits),
        inplace=True,
    )

    return combined_circuit


def circuit_to_qasm3_string(circuit: QuantumCircuit) -> str:
    combined_qasm3 = qasm3.dumps(circuit)
    combined_qasm3_cleaned = dedent(combined_qasm3).lstrip()
    return combined_qasm3_cleaned


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Loop.instance.identifier}.optimize_ansatz", bind=True)
def optimize_ansatz(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting the optmizing loop with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    task_options: Dict[str, Union[str, int]] = LoopParametersSchema().loads(
        task_data.parameters or "{}"
    )
    print(task_options)

    print("td", task_data)

    optimizer: Optional[int] = task_options["optimizer"]

    # Combine state prep and ansatz
    combined_circuit = combine_circuit(db_id=db_id)
    # print(combined_circuit)

    num_params = combined_circuit.num_parameters
    # print("Number of parameters:", num_params)
    TASK_LOGGER.info(f"Number of parameters: {num_params}")

    initial_guess = np.random.uniform(
        0, np.pi, num_params
    )  # NOTE another value range could perform better
    TASK_LOGGER.info(f"Initial guess: {initial_guess}")

    cost_fun = get_cost_function(db_id, combined_circuit)

    results = []
    # TODO adapt to work well with flask
    if optimizer == OPTIMIZERENUM.COBYLA:
        print("Starting optimization using COBYLA...")
        result = minimize(
            fun=cost_fun,
            method="COBYLA",
            x0=initial_guess,
            # options={"maxiter": 1},  # NOTE for quick results during debugging
        )
        results.append(result)

        print("Minimizer result:", result)
    elif optimizer == OPTIMIZERENUM.SPSA:
        # TODO
        pass

    x_values = [res.x.tolist() for res in results]
    x_json = json.dumps(x_values, indent=2)

    task_data.data["results"] = x_json

    task_data.save(commit=True)

    app = current_app._get_current_object()
    TASK_DETAILS_CHANGED.send(app, task_id=task_data.id)

    return self.replace(
        circuit_result_task.si(db_id=db_id) | save_task_result.s(db_id=db_id)
    )


@CELERY.task(name=f"{Loop.instance.identifier}.circuit_result_task", bind=True)
def circuit_result_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Saving circuit demo task results with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to save results!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    circuit_url = task_data.data.get("circuit_url")

    if circuit_url and isinstance(circuit_url, str):
        with open_url(circuit_url) as circuit_response:
            with SpooledTemporaryFile(mode="w") as output:
                output.write(circuit_response.text)
                STORE.persist_task_result(
                    db_id,
                    output,
                    "circuit.qasm",
                    "executable/circuit",
                    "text/x-qasm",
                )

    results = task_data.data.get("results")

    with SpooledTemporaryFile(mode="w") as output:
        output.write(results)
        STORE.persist_task_result(
            db_id,
            output,
            "circuit.qasm",
            "executable/circuit",
            "text/x-qasm",
        )

    return "Success"


    return "Successfully saved circuit executor task result!"
