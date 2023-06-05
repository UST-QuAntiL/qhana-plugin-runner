from typing import Optional

import numpy as np
import requests
from celery.utils.log import get_task_logger
from scipy.optimize import minimize as scipy_minimize

from ..coordinator.shared_schemas import (
    CalcLossInputData,
    CalcLossInputDataSchema,
    LossResponseData,
    LossResponseSchema,
    MinimizerInputData,
    MinimizerInputSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from . import Minimizer

TASK_LOGGER = get_task_logger(__name__)


def loss_(loss_calc_endpoint_url):
    def loss(x, y, x0, hyperparameters):
        request_schema = CalcLossInputDataSchema()
        request_data = request_schema.dump(
            CalcLossInputData(x0=x0, x=x, y=y, hyperparameters=hyperparameters)
        )

        response = requests.post(loss_calc_endpoint_url, json=request_data)
        response_schema = LossResponseSchema()
        response_data: LossResponseData = response_schema.load(response.json())
        return response_data.loss

    return loss


@CELERY.task(name=f"{Minimizer.instance.identifier}.minimize", bind=True)
def minimize_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting the optimization task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    method: str = task_data.data.get("method")
    input_data = {
        "x": task_data.data.get("x"),
        "y": task_data.data.get("y"),
        "hyperparameters": task_data.data.get("hyperparameters"),
        "calcLossEndpointUrl": task_data.data.get("calc_loss_endpoint_url"),
    }
    schema = MinimizerInputSchema()
    minimizer_input_data: MinimizerInputData = schema.load(input_data)
    loss_fun = loss_(minimizer_input_data.calc_loss_endpoint_url)

    initial_weights = np.random.randn(minimizer_input_data.x.shape[1])
    result = scipy_minimize(
        loss_fun,
        initial_weights,
        args=(
            minimizer_input_data.x,
            minimizer_input_data.y,
            minimizer_input_data.hyperparameters,
        ),
        method=method,
    )

    TASK_LOGGER.info(f"Optimization result: {result}")
    return ", ".join(map(str, result.x.flatten()))
