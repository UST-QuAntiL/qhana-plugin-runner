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
from typing import Optional

import requests
import numpy as np
from celery.utils.log import get_task_logger
from scipy.optimize import minimize as scipy_minimize

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
    ensure_array,
)
from qhana_plugin_runner.plugin_utils.interop import get_task_result_no_wait
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url, get_mimetype

from . import ScipyMinimizer

TASK_LOGGER = get_task_logger(__name__)


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
        response = requests.post(calc_loss_endpoint_url, json={"weights": weights})
        return response.json()["loss"]

    return loss


# TODO: remove (gradients not required here)
def jac_(calc_gradient_endpoint_url: str):
    """
    Function generator to calculate the gradient. This returns a function that calculates the gradient
    for given input data and hyperparameters.

    Args:
        calc_gradient_endpoint_url: The URL to which the gradient calculation request will be sent.

    Returns:
        A function that calculates the gradient.
    """

    def jac(x0):
        weights = x0
        if isinstance(x0, np.ndarray):
            weights = x0.tolist()

        response = requests.post(calc_gradient_endpoint_url, json={"weights": weights})
        return np.ndarray(response.json()["gradient"])

    return jac


# TODO: remove (gradients not required here)
def loss_and_jac_(calc_loss_and_gradient_endpoint_url: str):
    """
    Function generator to calculate the loss and gradient. This returns a function that calculates the loss
    and gradient for given input data and hyperparameters.

    Args:
        calc_loss_and_gradient_endpoint_url: The URL to which the loss and gradient calculation request will be sent.

    Returns:
        A function that calculates the loss and gradient.
    """

    def loss_and_jac(x0):
        weights = x0
        if isinstance(x0, np.ndarray):
            weights = x0.tolist()

        response = requests.post(
            calc_loss_and_gradient_endpoint_url, json={"weights": weights}
        )
        data = response.json()
        return data["loss"], np.ndarray(data["gradient"])

    return loss_and_jac


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

    nr_of_weights = weights_response.json()[
        "weights"
    ]  # FIXME load from objective function task

    if not isinstance(nr_of_weights, int) or nr_of_weights > 0:
        raise ValueError(
            f"Objective function provided a nonsensical nr of weights '{nr_of_weights}'!"
        )

    initial_weights = np.random.randn(
        nr_of_weights
    )  # FIXME load from initial weights or randomize from number of weights

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

    # TODO: remove (gradients not required here)
    """
    if minimizer_input_data.calc_gradient_endpoint_url:
        jac = jac_(minimizer_input_data.calc_gradient_endpoint_url)
        minimize_params["jac"] = jac

    if minimizer_input_data.calc_loss_and_gradient_endpoint_url:
        loss_and_jac = loss_and_jac_(
            minimizer_input_data.calc_loss_and_gradient_endpoint_url
        )
        minimize_params["jac"] = True
        minimize_params["fun"] = loss_and_jac
    """

    result = scipy_minimize(**minimize_params)

    csv_attributes = [f"x_{i}" for i in range(len(result.x))]

    final_weights = dict(zip(csv_attributes, result.x.tolist()))

    final_weights["ID"] = "weights"

    entities = [final_weights]

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(entities, output, "text/csv", attributes=csv_attributes)
        STORE.persist_task_result(
            db_id,
            output,
            "minimization-results.csv",
            "entity/vector",
            "text/csv",
        )
    return "Success"
