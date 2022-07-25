import json
from tempfile import SpooledTemporaryFile
from typing import Optional, Dict

import requests
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from . import ObjectiveFunctionDemo
from .neural_network import NeuralNetwork

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{ObjectiveFunctionDemo.instance.identifier}.setup_task", bind=True)
def setup_task(self, db_id: int) -> str:
    """
    Retrieves the input data from the database and stores metadata and hyperparameters into files.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting objective function setup task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    parameters: Dict = json.loads(task_data.parameters)
    number_of_input_values: int = parameters.get("number_of_input_values")
    number_of_neurons: int = parameters.get("number_of_neurons")
    callback_url: str = parameters.get("callback_url")

    TASK_LOGGER.info(
        f"Loaded data from db: number_of_input_values='{number_of_input_values}'"
    )
    TASK_LOGGER.info(f"Loaded data from db: number_of_neurons='{number_of_neurons}'")
    TASK_LOGGER.info(f"Loaded data from db: callback_url='{callback_url}'")

    if number_of_input_values is None or number_of_neurons is None:
        raise ValueError("Input parameters incomplete")

    model = NeuralNetwork(number_of_input_values, number_of_neurons)

    with SpooledTemporaryFile(mode="w") as output:
        output.write(task_data.parameters)
        STORE.persist_task_result(
            db_id,
            output,
            "hyperparameters.json",
            "objective-function-hyperparameters",
            "application/json",
        )

    requests.post(
        callback_url,
        json={"dbId": db_id, "numberOfParameters": model.get_number_of_parameters()},
    )

    return "Stored metadata and hyperparameters"
