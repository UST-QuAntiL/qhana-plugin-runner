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

from tempfile import SpooledTemporaryFile
from typing import Optional, Any
from time import sleep, time

import numpy as np
import requests
from requests.exceptions import ConnectionError, Timeout
from celery.utils.log import get_task_logger
from scipy.optimize import minimize as scipy_minimize

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_array,
    load_entities,
    save_entities,
    array_to_entity,
    ArrayEntity,
)
from qhana_plugin_runner.plugin_utils.interop import get_task_result_no_wait
from qhana_plugin_runner.requests import get_mimetype, open_url
from qhana_plugin_runner.storage import STORE

from . import ScipyMinimizer

TASK_LOGGER = get_task_logger(__name__)


def async_request(url: str, json: Optional[Any] = None, timeout: int = 24 * 6 * 60):
    """Call an evaluation endpoint of the objective function and return the result.

    This function follows redirects and polls the endpoint until the result is available if required.
    The function does an exponential backoff up to 30 seconds to reduce load on the target server.
    If the connection fails for some reason 5 times on succession, the connectione error will be escalated.

    Args:
        url (str): the url to call
        json (Optional[Any], optional): the data to pass to the endpoint. Defaults to None.
        timeout (int, optional): the timeout in seconds after which an error will be thrown. Defaults to 24 hours.

    Raises:
        Timeout: reached the final timeout

    Returns:
        Response: the resulting response
    """
    errors = 0
    is_first = True
    sleep_duration = 1
    max_sleep = 30
    timeout_after = time() + timeout
    while errors < 5:
        if time() > timeout_after:
            raise Timeout()  # timeout after specified time
        if not is_first:
            # sleep after first request and adjust next sleep duration
            sleep(sleep_duration)
            sleep_duration = min(max_sleep, sleep_duration * 2)
        try:
            response = requests.request(
                method=("POST" if is_first else "GET"), url=url, json=json, timeout=3
            )
            if is_first:
                url = response.url  # follow redirects on first request
            is_first = False
            errors = max(0, errors - 1)  # successfull attempts decrease errors
        except ConnectionError:
            is_first = False
            errors += 1
            if errors >= 5:
                raise
            continue  # error with the connection, wait and retry
        if response.status_code == 204 or (is_first and response.status_code == 404):
            continue  # no result available, wait and retry
        response.raise_for_status()
        return response


def loss_(calc_loss_endpoint_url: str):
    """
    Function generator to calculate loss. This returns a function that calculates loss
    for given input data and hyperparameters.

    Args:
        loss_calc_endpoint_url: The URL to which the loss calculation request will be sent.

    Returns:
        A function that calculates the loss.
    """

    def loss(x0):
        weights = x0
        if isinstance(x0, np.ndarray):
            weights = x0.tolist()
        response = async_request(url=calc_loss_endpoint_url, json={"weights": weights})
        return response.json()["loss"]

    return loss


@CELERY.task(name=f"{ScipyMinimizer.instance.identifier}.minimize", bind=True)
def minimize_task(self, db_id: int) -> str:
    """
    Start a minimization task for the specified database id.

    Args:
        db_id: The database id of the task.

    Returns:
        A string representation of the weights obtained from the minimization process.

    Raises:
        A KeyError if task data with the specified id cannot be loaded to read parameters.
    """
    TASK_LOGGER.info(f"Starting the optimization task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(task_data.data, dict)

    method: str = task_data.data.get("method")

    of_task_url = task_data.data["objective_function_task"]

    of_status, of_task_result = get_task_result_no_wait(of_task_url)

    if of_status != "PENDING":
        raise ValueError(f"Objective function task is in a wrong state '{of_status}'!")

    of_step_id = of_task_result["steps"][-1]["stepId"]
    if of_step_id != "evaluate":
        raise ValueError(
            f"Objective function task is not in the right step for minimization! (expected 'evaluate' but got '{of_step_id}'!"
        )

    calc_loss_endpoint = None
    get_weight_count_endpoint = None
    for link in of_task_result.get("links", []):
        if link["type"] == "of-weights":
            get_weight_count_endpoint = link["href"]
        elif link["type"] == "of-evaluate":
            calc_loss_endpoint = link["href"]

    if not calc_loss_endpoint:
        raise ValueError("Objective function task does not provide a 'of-evaluate' link!")
    if not get_weight_count_endpoint:
        raise ValueError("Objective function task does not provide a 'of-weights' link!")

    weights_response = requests.get(get_weight_count_endpoint, timeout=3)
    weights_response.raise_for_status()

    nr_of_weights = weights_response.json()["weights"]

    if not isinstance(nr_of_weights, int) or nr_of_weights <= 0:
        raise ValueError(
            f"Objective function provided a nonsensical nr of weights '{nr_of_weights}' ({type(nr_of_weights)})!"
        )

    initial_weights = np.random.randn(nr_of_weights)

    initial_weights_url = task_data.data.get("initial_weights_url", None)

    if initial_weights_url and isinstance(initial_weights_url, str):
        with open_url(initial_weights_url, stream=True) as initial_weights_response:
            mimetype = get_mimetype(initial_weights_response)
            if not mimetype:
                raise ValueError("Could not determine mimetype of y!")

            data = load_entities(initial_weights_response, mimetype=mimetype)
            array_data = ensure_array(data, strict=True)

            # get first entity data
            initial_weights = np.array(next(array_data).values)

        if len(initial_weights) != nr_of_weights:
            raise ValueError(
                f"Supplied initial weights have the wrong amount of weights! (Expected {nr_of_weights} but got {len(initial_weights)})"
            )

        if (0 > initial_weights).any() or (1 < initial_weights).any():
            raise ValueError("Initial weights may only have values between 0 and 1!")

    loss_fun = loss_(calc_loss_endpoint)

    minimize_params = {"fun": loss_fun, "x0": initial_weights, "method": method}

    result = scipy_minimize(**minimize_params)

    TASK_LOGGER.info(f"Optimization result: {result}")

    array_entities = [ArrayEntity("weights", "", result.x.tolist())]
    entities = tuple(array_to_entity(array_entities, prefix="x_"))

    csv_attributes = entities[0].entity_attributes

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entities, output, "text/csv", attributes=csv_attributes)
        STORE.persist_task_result(
            db_id,
            output,
            f"final_weights_scipy_{method}.csv",
            "entity/vector",
            "text/csv",
        )
    return "Success"
