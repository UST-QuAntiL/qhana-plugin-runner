import io
import json
import time
from json import loads
from tempfile import SpooledTemporaryFile
from typing import Optional

import numpy as np
from celery.utils.log import get_task_logger
from common.algorithms import are_vectors_orthogonal
from common.plugin_utils.plugin_handler import create_qasm_file_and_get_url
from common.plugin_utils.task_util import generate_one_circuit_with_two_states
from common.quantum_algorithms import (
    generate_swaptest_circuit,
    interpret_swaptest_results,
)
from flask.globals import current_app, request
from qiskit import QuantumCircuit

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.interop import (
    call_plugin_endpoint,
    get_plugin_endpoint,
    get_task_result_no_wait,
    monitor_external_substep,
    monitor_result,
    subscribe,
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import (
    TASK_DETAILS_CHANGED,
    TASK_STEPS_CHANGED,
    save_task_error,
    save_task_result,
)

from . import Plugin

TASK_LOGGER = get_task_logger(__name__)

_task1name_ = "building_circuit_and_simulate_task"
_task2name_ = "get_restults_task"
_task3name_ = "_interpret_restults_task"


@CELERY.task(
    name=f"{Plugin.instance.identifier}.{_task1name_}_task",
    bind=True,
)
def building_circuit_and_simulate_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting '{_task1name_}' task with DB ID='{db_id}'.")

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if not task_data:
        msg = f"No task data found for ID {db_id}."
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = loads(task_data.parameters or "{}")
    TASK_LOGGER.info(f"Parameters: {params}")

    ##
    # Get inputs
    execution_options_url = params.get("executionOptions", None)
    shots = params.get("shots", None)
    qasm_input_list = params.get("qasmInputList", None)
    executor = params.get("executor", None)

    try:
        # Input check
        if executor is None:
            task_data.add_task_log_entry(
                "No executor plugin specified, aborting task.", commit=True
            )
            raise ValueError(
                "Cannot execute a quantum circuit without a circuit executor plugin specified."
            )

        # Generate the QASM code and qubit intervals
        states_circuit, qubit_intervals = generate_one_circuit_with_two_states(
            qasm_input_list
        )

        # Ensure that exactly two states are present
        if len(qubit_intervals) != 2:
            raise ValueError("Exactly two states must be provided.")

        (first_start, first_end), (second_start, second_end) = qubit_intervals

        # Compute the sizes of the states
        first_size = first_end - first_start
        second_size = second_end - second_start

        # Check if both states have the same dimension
        if first_size != second_size:
            raise ValueError(
                f"To perform the orthogonality check, both states must have the same dimension. "
                f"Provided dimensions: {first_size} and {second_size}."
            )

        # Log original circuit as QASM
        TASK_LOGGER.info("Original circuit (QASM format):\n%s", states_circuit.qasm())
        TASK_LOGGER.info(f"Qubit intervals: {qubit_intervals}")

        # Generate and append the Swap Test
        swaptest_circuit = generate_swaptest_circuit(
            states_circuit, first_start, first_end, second_start, second_end
        )

        # Log swap test circuit as QASM
        TASK_LOGGER.info(
            "Circuit after appending the Swap Test (QASM format):\n%s",
            swaptest_circuit.qasm(),
        )

        # Stores the circut and gets url
        swaptest_circuit_url = create_qasm_file_and_get_url(swaptest_circuit.qasm())
        executorEndpoint = get_plugin_endpoint(executor)
        continue_url = task_data.data["continue_url"]

        # Prepare the payload based on the provided parameters.
        if shots and execution_options_url:
            payload = {
                "shots": shots,
                "circuit": swaptest_circuit_url,
                "executionOptions": execution_options_url,
            }
        elif shots:
            payload = {
                "shots": shots,
                "circuit": swaptest_circuit_url,
            }
        elif execution_options_url:
            payload = {
                "circuit": swaptest_circuit_url,
                "executionOptions": execution_options_url,
            }
        else:
            payload = {
                "circuit": swaptest_circuit_url,
            }

        result_url = call_plugin_endpoint(
            executorEndpoint,
            payload,
        )
        TASK_LOGGER.info(f"Task url: {result_url}")
        task_data.data["result_url"] = result_url
        task_data.save(commit=True)  # commit to save result url to DB

        subscribed = subscribe(
            result_url=result_url, webhook_url=continue_url, events=["steps", "status"]
        )
        task_data.data["subscribed"] = subscribed

        task_data.save(commit=True)

        app = current_app._get_current_object()
        TASK_DETAILS_CHANGED.send(app, task_id=task_data.id)

        if not subscribed:
            return self.replace(
                monitor_result.s(
                    result_url=result_url, webhook_url=continue_url, monitor="all"
                )
            )
        return

    except Exception as e:
        TASK_LOGGER.error(f"Error in '{_task1name_}' task: {e}")
        raise


@CELERY.task(name=f"{Plugin.instance.identifier}.{_task2name_}_task", bind=True)
def get_restults_task(self, db_id: int, event_type: Optional[str]):
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to save results!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    result_url = task_data.data.get("result_url")
    continue_url = task_data.data["continue_url"]
    subscribed = task_data.data["subscribed"]

    if result_url is None:
        raise ValueError(f"No result URL present in task data with id {db_id}")

    status, result = get_task_result_no_wait(result_url)

    if status == "FAILURE":
        if "--- Circuit Executor Log ---" not in task_data.task_log:
            task_data.add_task_log_entry(
                f"--- Circuit Executor Log ---\n{result.get('log', '')}\n--- END ---\n",
                commit=True,
            )
        raise ValueError("Circuit executor failed to execute the circuit!")
    elif status == "PENDING" and event_type != "status":
        steps = result.get("steps", [])
        external_step_id = add_new_substep(task_data, steps)
        if external_step_id and not subscribed:
            return self.replace(
                # wait for external substep to clear
                monitor_external_substep.s(
                    result_url=result_url,
                    webhook_url=continue_url,
                    substep=external_step_id,
                )
            )
        elif not subscribed:
            return self.replace(
                # wait for external substep or status change
                monitor_result.s(
                    result_url=result_url, webhook_url=continue_url, monitor="all"
                )
            )
    elif status == "SUCCESS" and event_type == "status":
        if "result" in task_data.data:
            return  # already checking for result, prevent duplicate task scheduling!

        task_data.data["result"] = result
        task_data.save(commit=True)

        return self.replace(
            _interpret_restults_task.si(db_id=db_id) | save_task_result.s(db_id=db_id)
        )
    else:
        raise ValueError(f"Unknown task status {status}!")


def add_new_substep(task_data: ProcessingTask, steps: list) -> Optional[int]:
    last_step = None
    if task_data.steps:
        last_step = task_data.steps[-1]
    current_step = steps[-1] if steps else None
    if current_step:
        step_id = current_step.get("stepId")
        if step_id:
            step_id = f"executor.{step_id}"
        else:
            step_id = f"executor.{len(steps)}"

        if not current_step.get("cleared", False):
            if (
                last_step
                and not last_step.cleared
                and last_step.step_id == step_id
                and last_step.href == current_step["href"]
            ):
                # new step and last step are identical, assume duplicate request and do nothing
                return None
            external_step_id = step_id if step_id else len(steps) - 1
            task_data.clear_previous_step()
            task_data.add_next_step(
                href=current_step["href"],
                ui_href=current_step["uiHref"],
                step_id=step_id,
                commit=True,
            )

            app = current_app._get_current_object()
            TASK_STEPS_CHANGED.send(app, task_id=task_data.id)

            return external_step_id
        elif current_step.get("cleared", False) and task_data.has_uncleared_step:
            if last_step and last_step.step_id == step_id:
                task_data.clear_previous_step(commit=True)
                app = current_app._get_current_object()
                TASK_STEPS_CHANGED.send(app, task_id=task_data.id)
    return None


@CELERY.task(name=f"{Plugin.instance.identifier}.{_task3name_}_task", bind=True)
def _interpret_restults_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Saving circuit demo task results with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to save results!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    circuit_url = task_data.data.get("circuit_url")
    if circuit_url and isinstance(circuit_url, str):
        STORE.persist_task_result(
            db_id,
            circuit_url,
            "circuit.qasm",
            "executable/circuit",
            "text/x-qasm",
            storage_provider="url_file_store",
        )

    # Find the URL for 'result-counts.json' in the task outputs.
    result_counts_url = None
    for output in task_data.data.get("result", {}).get("outputs", []):
        if output.get("name") == "result-counts.json":
            result_counts_url = output.get("href")
            break
    if not result_counts_url:
        raise ValueError("result-counts.json URL not found in task outputs.")
    # Fetch the result counts data.
    try:
        with open_url(result_counts_url) as counts_response:
            counts_response.raise_for_status()
            counts_data = counts_response.json()
    except Exception as e:
        raise RuntimeError(f"Error retrieving state vectors from QASM code: {e}")
    # Ensure that "0" and "1" exist in counts_data, setting them to 0 if missing
    counts_data.setdefault("0", 0)
    counts_data.setdefault("1", 0)

    # Interpret the results
    shots = counts_data["0"] + counts_data["1"]
    result = interpret_swaptest_results(counts_data["0"], counts_data["1"], shots)
    output_data = {"result": bool(result)}
    # Save results
    with SpooledTemporaryFile(mode="w") as json_file:
        json.dump(output_data, json_file)
        json_file.seek(0)
        STORE.persist_task_result(
            db_id,
            json_file,
            "out.json",
            f"custom/{_task1name_}-output",
            "application/json",
        )
    TASK_LOGGER.info(f"{_task1name_} result: {output_data}")
    return json.dumps(output_data)
