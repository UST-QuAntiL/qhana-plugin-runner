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
from celery.utils.log import get_task_logger
from scipy.optimize import minimize as scipy_minimize

from .shared.schemas import (
    CalcLossOrGradInput,
    CalcLossOrGradInputSchema,
    GradientResponseData,
    GradientResponseSchema,
    LossAndGradientResponseData,
    LossAndGradientResponseSchema,
    LossResponseData,
    LossResponseSchema,
    MinimizerInputData,
    MinimizerInputSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.storage import STORE

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
        request_data = CalcLossOrGradInputSchema().dump(CalcLossOrGradInput(x0=x0))

        response = requests.post(calc_loss_endpoint_url, json=request_data)
        response_data: LossResponseData = LossResponseSchema().load(response.json())
        return response_data.loss

    return loss


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
        request_data = CalcLossOrGradInputSchema().dump(CalcLossOrGradInput(x0=x0))

        response = requests.post(calc_gradient_endpoint_url, json=request_data)
        response_data: GradientResponseData = GradientResponseSchema().load(
            response.json()
        )
        return response_data.gradient

    return jac


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
        request_data = CalcLossOrGradInputSchema().dump(CalcLossOrGradInput(x0=x0))

        response = requests.post(calc_loss_and_gradient_endpoint_url, json=request_data)
        response_data: LossAndGradientResponseData = LossAndGradientResponseSchema().load(
            response.json()
        )
        return response_data.loss, response_data.gradient

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

    method: str = task_data.data.get("method")
    input_data = {
        "x0": task_data.data.get("x0"),
        "calcLossEndpointUrl": task_data.data.get("calc_loss_endpoint_url"),
        "calcGradientEndpointUrl": task_data.data.get("calc_gradient_endpoint_url"),
        "calcLossAndGradientEndpointUrl": task_data.data.get(
            "calc_loss_and_gradient_endpoint_url"
        ),
    }
    minimizer_input_data: MinimizerInputData = MinimizerInputSchema().load(input_data)
    loss_fun = loss_(minimizer_input_data.calc_loss_endpoint_url)

    minimize_params = {"fun": loss_fun, "x0": minimizer_input_data.x0, "method": method}

    if minimizer_input_data.calc_gradient_endpoint_url:
        jac = jac_(minimizer_input_data.calc_gradient_endpoint_url)
        minimize_params["jac"] = jac

    if minimizer_input_data.calc_loss_and_gradient_endpoint_url:
        loss_and_jac = loss_and_jac_(
            minimizer_input_data.calc_loss_and_gradient_endpoint_url
        )
        minimize_params["jac"] = True
        minimize_params["fun"] = loss_and_jac

    result = scipy_minimize(**minimize_params)

    csv_attributes = [f"x_{i}" for i in range(len(result.x))]

    entities = [dict(zip(csv_attributes, result.x.tolist()))]

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