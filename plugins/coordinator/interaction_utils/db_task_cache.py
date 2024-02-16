import sqlalchemy_json
from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)


def convert_tracked_to_standard(data):
    """Convert TrackedDict and TrackedList to standard dict and list."""
    if isinstance(data, sqlalchemy_json.track.TrackedDict):
        return {key: convert_tracked_to_standard(value) for key, value in data.items()}
    elif isinstance(data, sqlalchemy_json.track.TrackedList):
        return [convert_tracked_to_standard(item) for item in data]
    else:
        return data
