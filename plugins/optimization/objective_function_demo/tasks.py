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

import requests
from celery.utils.log import get_task_logger

from qhana_plugin_runner.api.plugin_schemas import (
    ObjectiveFunctionCallbackSchema,
    ObjectiveFunctionCallbackData,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from . import ObjectiveFunctionDemo
from .neural_network import NeuralNetwork
from .schemas import (
    HyperparametersSchema,
    InternalData,
    InternalDataSchema,
)

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{ObjectiveFunctionDemo.instance.identifier}.setup_task", bind=True)
def setup_task(self, db_id: int) -> str:
    """
    Retrieves the input data from the database and stores metadata and hyperparameters into files.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @param calculation_endpoint:
    @return: log message
    """
    TASK_LOGGER.info(f"Starting objective function setup task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    schema = InternalDataSchema()
    internal_data: InternalData = schema.loads(task_data.parameters)

    TASK_LOGGER.info(
        f"Loaded data from db: number_of_input_values='{internal_data.hyperparameters.number_of_input_values}'"
    )
    TASK_LOGGER.info(
        f"Loaded data from db: number_of_neurons='{internal_data.hyperparameters.number_of_neurons}'"
    )
    TASK_LOGGER.info(
        f"Loaded data from db: callback_url='{internal_data.callback_url.callback_url}'"
    )

    model = NeuralNetwork(
        internal_data.hyperparameters.number_of_input_values,
        internal_data.hyperparameters.number_of_neurons,
    )

    with SpooledTemporaryFile(mode="w") as output:
        output.write(HyperparametersSchema().dumps(internal_data.hyperparameters))
        STORE.persist_task_result(
            db_id,
            output,
            "hyperparameters.json",
            "objective-function-hyperparameters",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        output.write(
            json.dumps({"number_of_parameters": model.get_number_of_parameters()})
        )
        STORE.persist_task_result(
            db_id,
            output,
            "callback_data.json",
            "callback-data",
            "application/json",
        )

    return "Stored metadata and hyperparameters"


@CELERY.task(name=f"{ObjectiveFunctionDemo.instance.identifier}.callback_task", bind=True)
def callback_task(
    self, _, callback_url: str, calculation_endpoint: str, task_url: str
) -> None:

    callback_schema = ObjectiveFunctionCallbackSchema()
    callback_data = ObjectiveFunctionCallbackData(
        calculation_url=calculation_endpoint, task_url=task_url
    )

    TASK_LOGGER.info(callback_schema.dump(callback_data))

    resp = requests.post(
        callback_url,
        json=callback_schema.dump(callback_data),
    )

    if resp.status_code >= 400:
        TASK_LOGGER.error(
            f"{resp.request.url} {resp.status_code} {resp.reason} {resp.text}"
        )