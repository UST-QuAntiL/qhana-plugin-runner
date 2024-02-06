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

from json import dump
from tempfile import SpooledTemporaryFile
from typing import Optional
from uuid import uuid4

from celery import chain
from celery.utils.log import get_task_logger
from flask.globals import current_app
from qiskit import QuantumCircuit, execute
from qiskit.result.result import ExperimentResult, Result
from qiskit_ibm_provider import IBMJob
from qiskit_ibm_runtime import QiskitRuntimeService

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import TASK_STEPS_CHANGED, add_step, save_task_result

from . import QiskitExecutor
from .backend.qiskit_backends import get_backend_names, get_qiskit_backend
from .schemas import (
    CircuitParameters,
    CircuitParameterSchema,
)

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{QiskitExecutor.instance.identifier}.start_execution", bind=True)
def start_execution(self, db_id: int) -> str:
    # get parameters
    TASK_LOGGER.info(f"Starting new qiskit executor task with db id '{db_id}'")
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    circuit_params: CircuitParameters = CircuitParameterSchema().load(
        db_task.data["parameters"]
    )

    backend_list = None
    if (
        not circuit_params.ibmqToken
        or (backend_list := get_backend_names(circuit_params.ibmqToken)) is None
    ):
        # start the authentication task
        href = db_task.data["urls"]["authentication"]
        ui_href = db_task.data["urls"]["authentication_ui"]
        msg = "Started authentication task"
        if circuit_params.ibmqToken:
            msg += " (invalid IBMQ token)"
        self.replace(
            add_step.s(
                task_log=msg,
                db_id=db_task.id,
                step_id="authentication",
                href=href,
                ui_href=ui_href,
                prog_value=1,
                prog_target=3,
                prog_unit="steps",
            )
        )
        return msg

    db_task.data["backend_names"] = backend_list
    db_task.save(commit=True)

    backend = None
    if (
        not circuit_params.backend
        or (
            backend := get_qiskit_backend(
                circuit_params.backend, circuit_params.ibmqToken
            )
        )
        is None
    ):
        # start the backend selection task
        href = db_task.data["urls"]["backend_selection"]
        ui_href = db_task.data["urls"]["backend_selection_ui"]
        msg = "Started backend selection task"
        if circuit_params.backend:
            msg += " (invalid backend)"
        self.replace(
            add_step.s(
                task_log=msg,
                db_id=db_task.id,
                step_id="backend-selection",
                href=href,
                ui_href=ui_href,
                prog_value=2,
                prog_target=3,
                prog_unit="steps",
            )
        )
        return msg

    circuit_qasm: str
    with open_url(circuit_params.circuit) as quasm_response:
        circuit_qasm = quasm_response.text

    backend.shots = circuit_params.shots
    circuit = QuantumCircuit.from_qasm_str(circuit_qasm)

    TASK_LOGGER.info(f"Start execution with parameters: {str(circuit_params)}")

    job: IBMJob = execute(circuit, backend, shots=circuit_params.shots)

    db_task.data["job_id"] = job.job_id()
    db_task.clear_previous_step()
    db_task.save(commit=True)

    app = current_app._get_current_object()
    TASK_STEPS_CHANGED.send(app, task_id=db_id)

    # start the result watcher task
    task: chain = result_watcher.si(db_id=db_task.id) | save_task_result.s(
        db_id=db_task.id
    )
    task.link_error(save_task_result.s(db_id=db_task.id))
    task.apply_async()

    return "Started executing job"


class JobNotFinished(Exception):
    pass


@CELERY.task(
    name=f"{QiskitExecutor.instance.identifier}.result_watcher",
    bind=True,
    ignore_result=True,
    autoretry_for=(JobNotFinished,),
    retry_backoff=True,
    max_retries=None,
)
def result_watcher(self, db_id: int) -> str:
    # get parameters
    TASK_LOGGER.info(f"Starting new qiskit executor task with db id '{db_id}'")
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    job_id = db_task.data["job_id"]
    params: CircuitParameters = CircuitParameterSchema().load(db_task.data["parameters"])
    execution_options = db_task.data["options"]

    service = QiskitRuntimeService(token=params.ibmqToken, channel="ibm_quantum")
    job = service.job(job_id)

    if not job.in_final_state():
        raise JobNotFinished("Job not finished yet!")  # auto-retry task on this exception

    result: Result = job.result()
    if not result.success:
        # TODO better error
        raise ValueError("Circuit could not be executed!", result)

    experiment_result: ExperimentResult = result.results[0]
    extra_metadata = result.metadata

    time_taken = result.time_taken
    time_taken_execute = extra_metadata.get("time_taken_execute", time_taken)
    shots = experiment_result.shots
    if isinstance(shots, tuple):
        assert (
            len(shots) == 2
        ), "If untrue, check with qiskit documentation what has changed!"
        shots = abs(shots[-1] - shots[0])
    seed = experiment_result.seed_simulator

    metadata = {
        # trace ids (specific to IBM qiskit jobs)
        "jobId": result.job_id,
        "qobjId": result.qobj_id,
        # QPU/Simulator information
        "qpuType": "simulator" if "simulator" in result.backend_name else "qpu",
        "qpuVendor": "IBM",
        "qpuName": result.backend_name,
        "qpuVersion": result.backend_version,
        "seed": seed,  # only for simulators
        "shots": shots,
        # Time information
        "date": str(result.date),
        "timeTaken": time_taken,  # total job time
        "timeTakenIdle": 0,  # idle/waiting time
        "timeTakenQpu": time_taken,  # total qpu time
        "timeTakenQpuPrepare": time_taken - time_taken_execute,
        "timeTakenQpuExecute": time_taken_execute,
    }

    counts = result.get_counts()

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

    extra_execution_options = {
        "ID": experiment_id,
        "executorPlugin": execution_options.get("executorPlugin", [])
        + [QiskitExecutor.instance.identifier],
        "shots": metadata.get("shots", params.shots),
        "qpuType": metadata["qpuType"],
        "qpuVendor": metadata["qpuVendor"],
        "qpuName": metadata["qpuName"],
        "qpuVersion": metadata["qpuVersion"],
        "backend": params.backend,
    }

    if "seed" in metadata:
        extra_execution_options["seed"] = metadata["seed"]

    extra_execution_options.update(execution_options)

    with SpooledTemporaryFile(mode="w") as output:
        dump(extra_execution_options, output)
        STORE.persist_task_result(
            db_id,
            output,
            "execution-options.json",
            "provenance/execution-options",
            "application/json",
        )

    return "Finished executing circuit"
