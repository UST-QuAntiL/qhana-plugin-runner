from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

from . import Optimizer
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{Optimizer.instance.identifier}.objective_function_selection", bind=True
)
def objective_function_selection(self, db_id: int) -> str:
    """
    First processing task. Retrieves the input url from the database and stores it in a file.
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

    invoked_plugin = task_data.data.get("invoked_plugin")
    TASK_LOGGER.info(f"Loaded input parameters from db: called plugin='{invoked_plugin}'")

    if invoked_plugin is None:
        raise ValueError("No input data provided!")

    out_str = "Selected objective-function plugin: " + invoked_plugin

    if invoked_plugin is None:
        raise ValueError("No objective-function plugin provided!")

    # store data in file
    with SpooledTemporaryFile(mode="w") as output:
        output.write(out_str)
        STORE.persist_task_result(
            db_id,
            output,
            "optimizer_output_of_selection.txt",
            "hello-world-output",
            "text/plain",
        )

    return "result: " + repr(out_str)


@CELERY.task(name=f"{Optimizer.instance.identifier}.dataset_selection", bind=True)
def dataset_selection(self, db_id: int) -> str:
    """
    First processing task. Retrieves the input url from the database and stores it in a file.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting dataset selection task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_file_url = task_data.data.get("input_file_url")
    out_str = "Selected dataset url: " + input_file_url

    # store data in file
    with SpooledTemporaryFile(mode="w") as output:
        output.write(out_str)
        STORE.persist_task_result(
            db_id,
            output,
            "optimizer_output_data_selection.txt",
            "hello-world-output",
            "text/plain",
        )

    return "result: " + repr(out_str)
