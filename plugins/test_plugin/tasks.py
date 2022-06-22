from tempfile import SpooledTemporaryFile

from typing import Optional
from json import loads

from celery.utils.log import get_task_logger

from . import Test
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Test.instance.identifier}.preprocessing_task", bind=True)
def preprocessing_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting preprocessing demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_str = task_data.data["input_str"]
    TASK_LOGGER.info(f"Loaded input parameters from db: input_str='{input_str}'")
    if input_str is None:
        raise ValueError("No input argument provided!")

    if input_str:
        out_str = "Processed in the preprocessing step: " + input_str
        with SpooledTemporaryFile(mode="w") as output:
            output.write(out_str)
            STORE.persist_task_result(
                db_id, output, "output1.txt", "hello-world-output", "text/plain"
            )
        return "result: " + repr(out_str)
    return "Empty input string, no output could be generated!"


@CELERY.task(name=f"{Test.instance.identifier}.processing_task", bind=True)
def processing_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting preprocessing demo task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_str: str = task_data.data["input_str"]
    TASK_LOGGER.info(f"Loaded input parameters from db: input_str='{input_str}'")
    if input_str is None:
        raise ValueError("No input argument provided!")

    if input_str:
        out_str = input_str

        with SpooledTemporaryFile(mode="w") as output:
            output.write(out_str)
            STORE.persist_task_result(
                db_id, output, "output2.txt", "hello-world-output", "text/plain"
            )
        return "result: " + repr(out_str)
    return "Empty input string, no output could be generated!"
