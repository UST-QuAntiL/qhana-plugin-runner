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
import json
from tempfile import SpooledTemporaryFile
from typing import Optional
from urllib.parse import urljoin

import requests
from celery.utils.log import get_task_logger

from qhana_plugin_runner.api.plugin_schemas import PluginMetadataSchema, PluginMetadata
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from . import OptimizationCoordinator

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{OptimizationCoordinator.instance.identifier}.no_op_task", bind=True)
def no_op_task(self, db_id: int) -> str:
    """
    First and second processing task. Does nothing.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting no op task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    return ""


@CELERY.task(
    name=f"{OptimizationCoordinator.instance.identifier}.start_optimization_task",
    bind=True,
)
def start_optimization_task(self, db_id: int) -> str:
    """
    Third processing task. Starts the optimization.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting optimization task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    objective_function_plugin_url = task_data.data["objective_function_url"]
    optimizer_plugin_url = task_data.data["optimizer_url"]
    dataset_url = task_data.data["dataset_url"]
    optim_db_id = task_data.data["optim_db_id"]
    number_of_parameters = task_data.data["number_of_parameters"]
    obj_func_db_id = task_data.data["obj_func_db_id"]
    obj_func_calc_url = _get_calc_endpoint(objective_function_plugin_url)

    schema = PluginMetadataSchema()
    plugin_metadata: PluginMetadata = schema.loads(
        requests.get(optimizer_plugin_url).text
    )
    optimizer_start_url: Optional[str] = None

    for entry_point in plugin_metadata.entry_point.interaction_endpoints:
        if entry_point.type == "start-optimization":
            optimizer_start_url = urljoin(objective_function_plugin_url, entry_point.href)

    if optimizer_start_url is None:
        raise ValueError("No interaction endpoint found with type start-optimization")

    request_data = {
        "dataset": dataset_url,
        "optimizerDbId": optim_db_id,
        "numberOfParameters": number_of_parameters,
        "objFuncDbId": obj_func_db_id,
        "objFuncCalcUrl": obj_func_calc_url,
    }
    res = requests.post(
        optimizer_start_url,
        json=request_data,
    ).json()

    with SpooledTemporaryFile(mode="w") as output:
        output.write(
            json.dumps(
                {
                    "last_objective_value": res["lastObjectiveValue"],
                    "optimized_parameters": res["optimizedParameters"],
                }
            )
        )
        STORE.persist_task_result(
            db_id, output, "result.json", "optimization-result", "application/json"
        )

    return ""


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
