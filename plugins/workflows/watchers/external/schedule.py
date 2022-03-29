from .external_task_watcher import camunda_task_watcher
from qhana_plugin_runner.celery import CELERY
from ...config import EXTERNAL_WATCHER_INTERVAL


@CELERY.on_after_finalize.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    Setup periodic tasks for the camunda external task watcher. This will create a new entry in the beat scheduler.
    For this to work the celery beat should be running.
    """
    sender.add_periodic_task(EXTERNAL_WATCHER_INTERVAL, camunda_task_watcher.s(), name='periodic task workflows watcher')
