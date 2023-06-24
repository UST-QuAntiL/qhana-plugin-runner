from plugins.optimizer.interaction_utils.signal_handlers import (
    task_status_changed_handler,
)
from qhana_plugin_runner.tasks import TASK_STATUS_CHANGED

# connect signal handlers
TASK_STATUS_CHANGED.connect(task_status_changed_handler)
