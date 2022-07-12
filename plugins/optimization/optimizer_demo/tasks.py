from json import dumps, loads
from tempfile import SpooledTemporaryFile

from typing import Optional
from urllib.parse import urljoin

import numpy as np
import requests
from celery.utils.log import get_task_logger
from scipy.optimize import minimize

from qhana_plugin_runner.api.plugin_schemas import PluginMetadata, PluginMetadataSchema
from . import OptimizerDemo
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile

from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{OptimizerDemo.instance.identifier}.processing_task_1", bind=True)
def processing_task_1(self, db_id: int) -> str:
    """
    First processing task. Retrieves the input data from the database and stores it in a file.
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
    dataset_url: str, hyperparameters_url: str, objective_function_calculation_url: str
):
    def objective_function(x: np.ndarray) -> float:
        request_data = {
            "dataSet": dataset_url,
            "hyperparameters": hyperparameters_url,
            "parameters": x.tolist(),
        }
        res = requests.post(
            objective_function_calculation_url,
            json=request_data,
        ).json()

        return res["objectiveValue"]

    return objective_function


@CELERY.task(name=f"{OptimizerDemo.instance.identifier}.processing_task_2", bind=True)
def processing_task_2(self, db_id: int) -> str:
    """
    Second processing task. Retrieves the input data from the database and stores it in a file.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting processing step 2 task with db id '{db_id}'")
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    objective_function_plugin_url = db_task.data.get("objective_function_url")

    if objective_function_plugin_url is None:
        raise ValueError("Objective function plugin URL missing")

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

    metadata_url = db_task.data.get("metadata_url")
    hyperparameter_url = db_task.data.get("hyperparameter_url")
    dataset_url = loads(db_task.parameters).get("dataset_url")

    metadata = requests.get(metadata_url).json()
    number_of_parameters = metadata["number_of_parameters"]
    parameters = np.random.normal(size=(number_of_parameters,))
    obj_func = objective_function_wrapper(
        dataset_url, hyperparameter_url, objective_function_calculation_url
    )

    result = minimize(obj_func, parameters, method="COBYLA")
    optimized_parameters: np.ndarray = result.x

    # store data in file
    with SpooledTemporaryFile(mode="w") as output:
        output.write(
            dumps(
                {
                    "optimized_parameters": optimized_parameters.tolist(),
                    "last_objective_value": obj_func(optimized_parameters),
                }
            )
        )
        STORE.persist_task_result(
            db_id, output, "output_step_2.json", "optimization-output", "application/json"
        )

    return "result: "
