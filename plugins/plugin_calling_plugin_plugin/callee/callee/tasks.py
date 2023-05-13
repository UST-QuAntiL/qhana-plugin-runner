from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

from . import Callee
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Callee.instance.identifier}.demo_task", bind=True)
def demo_task(self, db_id: int) -> str:
    """
    Demo processing task. Retrieves the input data from the database and stores it in a file.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting new invoked task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_str: str = task_data.data.get("input_str")
    TASK_LOGGER.info(f"Loaded input data from db: input_str='{input_str}'")

    if input_str is None:
        raise ValueError("No input data provided!")

    out_str = "User input from invoked plugin micro frontend: " + input_str
    with SpooledTemporaryFile(mode="w") as output:
        output.write(out_str)
        STORE.persist_task_result(
            db_id,
            output,
            "output_callee.txt",
            "hello-world-output",
            "text/plain",
        )
    return "result: " + repr(out_str)


@CELERY.task(name=f"{Callee.instance.identifier}.demo_task_2", bind=True)
def demo_task_2(self, db_id: int) -> str:
    """
    Demo processing task. We already showed how to retrieve the input data from the database in the first demo task.
    Here we just show that a second task in the invoked plugin can be called.
    @param self:
    @param db_id: database ID that will be used to retrieve the task data from the database
    @return: log message
    """
    TASK_LOGGER.info(f"Starting second invoked task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    with SpooledTemporaryFile(mode="w") as output:
        out_str = "This is the second invoked task."
        output.write(out_str)
        STORE.persist_task_result(
            db_id,
            output,
            "output_callee_step_2.txt",
            "hello-world-output",
            "text/plain",
        )
    return "result: " + repr(out_str)
