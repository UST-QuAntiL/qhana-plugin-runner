from qhana_plugin_runner.celery import CELERY

from .external_task_watcher import camunda_task_watcher
from ... import Workflows

CELERY.add_periodic_task( # FIXME is this the right place??
    Workflows.instance.config["polling_rates"]["external_watcher"],
    camunda_task_watcher.s(),
    name="periodic task workflows watcher",
)

print("\n\n\n", CELERY.conf.humanize(with_defaults=False, censored=True), "\n\n\n") # TODO remove


@CELERY.on_after_finalize.connect  # FIXME signal already fired???
def setup_periodic_tasks(sender, **kwargs):
    """
    Setup periodic tasks for the camunda external task watcher. This will create a new entry in the beat scheduler.
    For this to work the celery beat should be running.
    """
    sender.add_periodic_task(
        Workflows.instance.config["polling_rates"]["external_watcher"],
        camunda_task_watcher.s(),
        name="periodic task workflows watcher",
    )
