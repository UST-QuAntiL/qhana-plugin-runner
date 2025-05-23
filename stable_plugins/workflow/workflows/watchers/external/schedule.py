from qhana_plugin_runner.celery import CELERY

from .external_task_watcher import camunda_task_watcher
from ... import DeployWorkflow


@CELERY.on_after_finalize.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    Setup periodic tasks for the camunda external task watcher. This will create a new entry in the beat scheduler.
    For this to work the celery beat should be running.
    """
    sender.add_periodic_task(
        DeployWorkflow.instance.config["camunda_queue_polling_rate"],
        camunda_task_watcher.s(),
        name="periodic task workflows watcher",
    )
