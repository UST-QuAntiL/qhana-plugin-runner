from json import dumps
from tempfile import SpooledTemporaryFile

from typing import Optional

import numpy as np
import requests
from celery.utils.log import get_task_logger
from scipy.optimize import minimize

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

    input_str = task_data.data.get("input_str")  # get data from database
    TASK_LOGGER.info(f"Loaded input parameters from db: input_str='{input_str}'")

    if input_str is None:
        raise ValueError("No input data provided!")

    out_str = "User input from step 1 micro frontend: " + input_str

    # store data in file
    with SpooledTemporaryFile(mode="w") as output:
        output.write(out_str)
        STORE.persist_task_result(
            db_id, output, "output_step_1.txt", "hello-world-output", "text/plain"
        )

    return "result: " + repr(out_str)


def objective_function_wrapper(data_set_url: str, hyperparameters_url: str):
    def objective_function(x: np.ndarray) -> float:
        request_data = {
            "dataSet": data_set_url,
            "hyperparameters": hyperparameters_url,
            "parameters": x.tolist(),
        }
        res = requests.post(
            "http://localhost:5005/plugins/objective-function-demo%40v0-1-0/calc/",
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

    metadata_url = db_task.data.get("metadata_url")
    hyperparameter_url = db_task.data.get("hyperparameter_url")

    metadata = requests.get(metadata_url).json()
    number_of_parameters = metadata["number_of_parameters"]
    parameters = np.random.normal(size=(number_of_parameters,))
    obj_func = objective_function_wrapper(
        "http://localhost:9090/experiments/1/data/dataset_test.json/download?version=1",
        hyperparameter_url,
    )  # FIXME: replace hardcoded URLs

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
            db_id, output, "output_step_2.txt", "optimization-output", "application/json"
        )

    return "result: "
