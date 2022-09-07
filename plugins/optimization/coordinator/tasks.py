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
import time
from tempfile import SpooledTemporaryFile
from typing import Optional
from urllib.parse import urljoin

import requests
from celery.utils.log import get_task_logger

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    OptimizationInput,
    OptimizationInputSchema,
    OptimizationOutputSchema,
    OptimizationOutput,
)
from qhana_plugin_runner.api.tasks_api import TaskStatusSchema, TaskStatus
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from . import OptimizationCoordinator
from .schemas import InternalDataSchema, InternalData

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
    TASK_LOGGER.info(f"Coordinator starting optimization task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    schema = InternalDataSchema()
    internal_data: InternalData = schema.loads(task_data.parameters)

    request_schema = OptimizationInputSchema()
    request_data = OptimizationInput(
        dataset=internal_data.dataset_url,
        number_of_parameters=internal_data.number_of_parameters,
        objective_function_calculation_url=internal_data.objective_function_calculation_url,
    )

    resp = requests.post(
        internal_data.optimizer_start_url,
        json=request_schema.dump(request_data),
    )

    if resp.status_code >= 400:
        TASK_LOGGER.error(
            f"{resp.request.url} {resp.status_code} {resp.reason} {resp.text}"
        )

    task_url = resp.url
    task_status_schema = TaskStatusSchema()
    optimization_output_url: Optional[str] = None

    while True:
        resp = requests.get(task_url)

        if resp.status_code >= 400:
            TASK_LOGGER.error(
                f"{resp.request.url} {resp.status_code} {resp.reason} {resp.text}"
            )

        task_status: TaskStatus = task_status_schema.load(resp.json())

        TASK_LOGGER.info(f"task status of optimizer plugin: {task_status.status}")

        if task_status.status == "SUCCESS":
            for output in task_status.outputs:
                if output.data_type == "optimization-output":
                    optimization_output_url = output.href
            break
        else:
            time.sleep(1)

    resp = requests.get(optimization_output_url)

    if resp.status_code >= 400:
        TASK_LOGGER.error(
            f"{resp.request.url} {resp.status_code} {resp.reason} {resp.text}"
        )

    with SpooledTemporaryFile(mode="w") as output:
        output.write(resp.text)
        STORE.persist_task_result(
            db_id, output, "result.json", "optimization-result", "application/json"
        )

    return ""
