from typing import Optional
from functools import lru_cache

import sqlalchemy_json
from celery.utils.log import get_task_logger

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask

TASK_LOGGER = get_task_logger(__name__)


@lru_cache
def get_of_calc_data(db_id: int) -> tuple:
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # Detach the entire db_task object from the session
    DB.session.expunge(db_task)

    # Convert TrackedDict and TrackedList to standard Python dict and list
    x = convert_tracked_to_standard(db_task.data["x"])
    y = convert_tracked_to_standard(db_task.data["y"])
    hyperparameter = convert_tracked_to_standard(db_task.data["hyperparameter"])

    return x, y, hyperparameter


def convert_tracked_to_standard(data):
    """Convert TrackedDict and TrackedList to standard dict and list."""
    if isinstance(data, sqlalchemy_json.track.TrackedDict):
        return {key: convert_tracked_to_standard(value) for key, value in data.items()}
    elif isinstance(data, sqlalchemy_json.track.TrackedList):
        return [convert_tracked_to_standard(item) for item in data]
    else:
        return data
