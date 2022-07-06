from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

from . import InteractionDemo
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{InteractionDemo.instance.identifier}.processing_task_1", bind=True)
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


@CELERY.task(name=f"{InteractionDemo.instance.identifier}.processing_task_2", bind=True)
def processing_task_2(self, db_id: int) -> str:
    """
    Second processing task. Retrieves the input data from the database and stores it in a file.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting processing step 2 task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_str: str = task_data.data.get("input_str")  # get data from database
    TASK_LOGGER.info(f"Loaded input data from db: input_str='{input_str}'")

    if input_str is None:
        raise ValueError("No input data provided!")

    out_str = "User input from step 2 micro frontend: " + input_str

    # store data in file
    with SpooledTemporaryFile(mode="w") as output:
        output.write(out_str)
        STORE.persist_task_result(
            db_id, output, "output_step_2.txt", "hello-world-output", "text/plain"
        )

    return "result: " + repr(out_str)
