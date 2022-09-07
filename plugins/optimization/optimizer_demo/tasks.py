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
from tempfile import SpooledTemporaryFile
from typing import Optional, List

import numpy as np
import requests
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from scipy.optimize import minimize

from qhana_plugin_runner.api.plugin_schemas import (
    OptimizerCallbackData,
    OptimizerCallbackSchema,
    OptimizationOutput,
    OptimizationOutputSchema,
    ObjFuncCalcInput,
    ObjFuncCalcInputSchema,
    ObjFuncCalcOutputSchema,
    ObjFuncCalcOutput,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from . import OptimizerDemo
from .schemas import InternalDataSchema, InternalData

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{OptimizerDemo.instance.identifier}.setup_task", bind=True)
def setup_task(self, db_id: int) -> str:
    """
    Retrieves the input data from the database and stores metadata and hyperparameters into files.

    :param self:
    :param db_id: database ID that will be used to retrieve the task data from the database
    :param optimizer_start_url: URL to the optimization endpoint
    :return: log message
    """
    TASK_LOGGER.info(f"Starting setup task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    schema = InternalDataSchema()
    parameters: InternalData = schema.loads(task_data.parameters)

    TASK_LOGGER.info(
        f"Loaded data from db: optimizer='{parameters.hyperparameters.optimizer}'"
    )

    if parameters.hyperparameters is None or parameters.callback_url is None:
        raise ValueError("Input parameters incomplete")

    with SpooledTemporaryFile(mode="w") as output:
        output.write(task_data.parameters)
        STORE.persist_task_result(
            db_id,
            output,
            "hyperparameters.json",
            "objective-function-hyperparameters",
            "application/json",
        )

    return "Stored metadata and hyperparameters"


@CELERY.task(name=f"{OptimizerDemo.instance.identifier}.callback_task", bind=True)
def callback_task(self, _, callback_url: str, task_url: str) -> None:
    TASK_LOGGER.info("Starting callback task")

    callback_schema = OptimizerCallbackSchema()
    callback_data = OptimizerCallbackData(task_url=task_url)

    resp = requests.post(callback_url, json=callback_schema.dump(callback_data))

    if resp.status_code >= 400:
        TASK_LOGGER.error(
            f"{resp.request.url} {resp.status_code} {resp.reason} {resp.text}"
        )

    # TODO: add comment why this line is necessary
    AsyncResult(self.request.parent_id, app=CELERY).forget()


@CELERY.task(name=f"{OptimizerDemo.instance.identifier}.optimization_task", bind=True)
def optimization_task(self, db_id: int) -> str:
    """
    Does the optimization.

    :param self:
    :param db_id: database ID that will be used to retrieve the task data from the database
    :return: log message
    """
    TASK_LOGGER.info(f"Starting optimization task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    schema = InternalDataSchema()
    internal_data: InternalData = schema.loads(task_data.parameters)

    # randomly initialize parameters
    parameters = np.random.normal(
        size=(internal_data.optimization_input.number_of_parameters,)
    )

    obj_func = _objective_function_wrapper(
        internal_data.optimization_input.dataset,
        internal_data.optimization_input.objective_function_calculation_url,
    )

    # optimization
    result = minimize(obj_func, parameters, method="COBYLA")

    # get results
    optimized_parameters: np.ndarray = result.x
    last_objective_value = obj_func(optimized_parameters)
    parameter_list: List[float] = optimized_parameters.tolist()

    output = OptimizationOutput(
        last_objective_value=last_objective_value, optimized_parameters=parameter_list
    )

    with SpooledTemporaryFile(mode="w") as file:
        file.write(OptimizationOutputSchema().dumps(output))
        STORE.persist_task_result(
            db_id,
            file,
            "optimization-output.json",
            "optimization-output",
            "application/json",
        )

    return "Stored metadata and hyperparameters"


def _objective_function_wrapper(
    dataset_url: str, objective_function_calculation_url: str
):
    """
    Provides the objective function with additional information that are needed to execute the requests to the objective
    function plugins.

    :param dataset_url:
    :param objective_function_calculation_url:
    :param obj_func_db_id:
    :return: the wrapped objective function that can be used with scipy optimizers
    """

    def objective_function(x: np.ndarray) -> float:
        request_data = ObjFuncCalcInput(data_set=dataset_url, parameters=x.tolist())
        input_schema = ObjFuncCalcInputSchema()

        resp = requests.post(
            objective_function_calculation_url,
            json=input_schema.dump(request_data),
        )

        if resp.status_code >= 400:
            TASK_LOGGER.error(
                f"{resp.request.url} {resp.status_code} {resp.reason} {resp.text}"
            )

        output_schema = ObjFuncCalcOutputSchema()
        output: ObjFuncCalcOutput = output_schema.load(resp.json())

        return output.objective_value

    return objective_function
