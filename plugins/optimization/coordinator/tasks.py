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
from json import dumps, loads
from tempfile import SpooledTemporaryFile
from typing import Optional
from urllib.parse import urljoin

import numpy as np
import requests
from celery.utils.log import get_task_logger
from scipy.optimize import minimize

from qhana_plugin_runner.api.plugin_schemas import PluginMetadata, PluginMetadataSchema
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from . import OptimizationCoordinator

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{OptimizationCoordinator.instance.identifier}.no_op_task", bind=True)
def no_op_task(self, db_id: int) -> str:
    """
    First processing task. Does nothing.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting processing step 1 task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    return ""


def objective_function_wrapper(
    dataset_url: str, objective_function_calculation_url: str, obj_func_db_id: int
):
    """
    Provides the objective function with additional informations that are needed to execute the requests to the objective function plugins.

    :param dataset_url:
    :param objective_function_calculation_url:
    :param obj_func_db_id:
    :return: the wrapped objective function that can be used with scipy optimizers
    """

    def objective_function(x: np.ndarray) -> float:
        request_data = {
            "dataSet": dataset_url,
            "dbId": obj_func_db_id,
            "parameters": x.tolist(),
        }
        res = requests.post(
            objective_function_calculation_url,
            json=request_data,
        ).json()

        return res["objectiveValue"]

    return objective_function


def _get_calc_endpoint(objective_function_plugin_url: str) -> str:
    schema = PluginMetadataSchema()
    plugin_metadata: PluginMetadata = schema.loads(
        requests.get(objective_function_plugin_url).text
    )
    objective_function_calculation_url: Optional[str] = None

    for entry_point in plugin_metadata.entry_point.interaction_endpoints:
        if entry_point.type == "objective-function-calculation":
            objective_function_calculation_url = urljoin(
                objective_function_plugin_url, entry_point.href
            )

    if objective_function_calculation_url is None:
        raise ValueError(
            "No interaction endpoint found in plugin with type objective-function-calculation"
        )

    return objective_function_calculation_url


@CELERY.task(
    name=f"{OptimizationCoordinator.instance.identifier}.optimization_task", bind=True
)
def optimization_task(self, db_id: int) -> str:
    """
    Second processing task. Uses an optimizer to optimize the specified objective function.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting optimization task with db id '{db_id}'")
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    objective_function_plugin_url = db_task.data.get("objective_function_url")

    if objective_function_plugin_url is None:
        raise ValueError("Objective function plugin URL missing")

    objective_function_calculation_url = _get_calc_endpoint(objective_function_plugin_url)

    obj_func_db_id = db_task.data.get("obj_func_db_id")
    dataset_url = loads(db_task.parameters).get("dataset_url")
    number_of_parameters = db_task.data.get("number_of_parameters")

    # randomly initialize parameters
    parameters = np.random.normal(size=(number_of_parameters,))

    obj_func = objective_function_wrapper(
        dataset_url, objective_function_calculation_url, obj_func_db_id
    )

    # optimization
    result = minimize(obj_func, parameters, method="COBYLA")

    # get results
    optimized_parameters: np.ndarray = result.x
    last_objective_value = obj_func(optimized_parameters)

    # store results in file
    with SpooledTemporaryFile(mode="w") as output:
        output.write(
            dumps(
                {
                    "optimized_parameters": optimized_parameters.tolist(),
                    "last_objective_value": last_objective_value,
                }
            )
        )
        STORE.persist_task_result(
            db_id, output, "output_step_2.json", "optimization-output", "application/json"
        )

    return f"result: {last_objective_value}"
