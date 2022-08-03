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
from typing import Optional

import requests
from celery.utils.log import get_task_logger

from qhana_plugin_runner.api.plugin_schemas import (
    OptimizerCallbackData,
    OptimizerCallbackSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from . import OptimizerDemo
from .schemas import InternalDataSchema, InternalData

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{OptimizerDemo.instance.identifier}.setup_task", bind=True)
def setup_task(self, db_id: int, optimizer_start_url: str) -> str:
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
    TASK_LOGGER.info(
        f"Loaded data from db: callback_url='{parameters.callback_url.callback_url}'"
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

    callback_schema = OptimizerCallbackSchema()
    callback_data = OptimizerCallbackData(optimizer_start_url=optimizer_start_url)

    resp = requests.post(
        parameters.callback_url.callback_url,
        json=callback_schema.dump(callback_data),
    )

    if resp.status_code >= 400:
        TASK_LOGGER.error(f"{resp.status_code} {resp.reason} {resp.text}")

    return "Stored metadata and hyperparameters"
