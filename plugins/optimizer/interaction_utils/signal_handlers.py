from requests import post
from qhana_plugin_runner.db.models.tasks import ProcessingTask


def task_status_changed_handler(sender, db_id: int):
    """Emit signal to all registered callback urls that the status of a task has changed.
    The signal contains the url of the task and the new status.
    Attributes:
        sender: The sender of the signal.
        db_id (int): The database id of the task.
    """
    task_data: ProcessingTask = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        raise KeyError(
            f"Could not find db entry for id {db_id}, failed to emit signal on status change!"
        )

    callback_urls = task_data.data.get("status_changed_callback_urls", [])
    task_url = task_data.data.get("task_view", None)
    for callback_url in callback_urls:
        post(callback_url, json={"url": task_url, "status": task_data.status})
