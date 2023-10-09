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

from json import dump, dumps, loads
import mimetypes
import os
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, Optional
from uuid import uuid4
from qiskit.providers.ibmq.job import IBMQJob
from qiskit.result.result import ExperimentResult, Result
from qiskit import QuantumCircuit, execute
from qiskit_ibm_runtime import QiskitRuntimeService
from celery.utils.log import get_task_logger

from .backend.qiskit_backends import get_qiskit_backend
from . import QiskitExecutor
from .schemas import (
    CircuitSelectionInputParameters,
    CircuitSelectionParameterSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    load_entities,
    ensure_dict,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{QiskitExecutor.instance.identifier}.prepare_task", bind=True)
def prepare_task(self, db_id: int) -> str:
    TASK_LOGGER.info(
        f"Starting new qiskit executor preparation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: CircuitSelectionInputParameters = (
        CircuitSelectionParameterSchema().loads(task_data.data)
    )
    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    # Save input data in internal data structure for further processing
    task_data.data = CircuitSelectionParameterSchema().dumps(input_params)

    task_data.save(commit=True)

    return f"Saved input data in internal data structure for further processing: {str(input_params)}"


@CELERY.task(name=f"{QiskitExecutor.instance.identifier}.start_execution", bind=True)
def start_execution(self, db_id: int) -> str:
    # get parameters
    TASK_LOGGER.info(f"Starting new qiskit executor task with db id '{db_id}'")
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    circuit_params: CircuitSelectionInputParameters = (
        CircuitSelectionParameterSchema().loads(db_task.data)
    )

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(circuit_params)}")

    circuit_qasm: str
    with open_url(circuit_params.circuit) as quasm_response:
        circuit_qasm = quasm_response.text

    execution_options: Dict[str, Any] = {
        "shots": circuit_params.shots,
    }

    if circuit_params.executionOptions:
        with open_url(circuit_params.executionOptions) as execution_options_response:
            try:
                mimetype = execution_options_response.headers["Content-Type"]
            except KeyError:
                mimetype = mimetypes.MimeTypes().guess_type(
                    url=circuit_params.executionOptions
                )[0]
            if mimetype is None:
                msg = "Could not guess execution options mime type!"
                TASK_LOGGER.error(msg)
                raise ValueError(msg)  # TODO better error
            entities = ensure_dict(
                load_entities(execution_options_response, mimetype=mimetype)
            )
            options = next(entities, {})
            db_task.add_task_log_entry(
                "loaded execution options: " + dumps(options), commit=True
            )
            execution_options.update(options)

    if circuit_params.ibmqToken == "****":
        TASK_LOGGER.info("Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            circuit_params.ibmqToken = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info("IBMQ_TOKEN environment variable not set")

    backend = get_qiskit_backend(circuit_params.backend, circuit_params.ibmqToken)
    if backend is None:
        msg = f"Could not load backend {circuit_params.backend}!"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)
    backend.shots = circuit_params.shots

    circuit = QuantumCircuit.from_qasm_str(circuit_qasm)

    job: IBMQJob = execute(circuit, backend, shots=execution_options["shots"])

    db_task.data = dumps(
        {
            "job_id": job.job_id(),
            "parameters": CircuitSelectionParameterSchema().dumps(circuit_params),
            "execution_options": execution_options,
        }
    )
    db_task.clear_previous_step()
    db_task.save(commit=True)

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

    data = loads(db_task.data)
    job_id = data["job_id"]
    params: CircuitSelectionInputParameters = CircuitSelectionParameterSchema().loads(
        data["parameters"]
    )
    execution_options = data["execution_options"]

    service = QiskitRuntimeService(token=params.ibmqToken, channel="ibm_quantum")
    job = service.job(job_id)

    if not job.in_final_state():
        raise JobNotFinished("Job not finished yet!")

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

    return "Finished executing circuit"
