from time import time
from typing import Optional

from celery.utils.log import get_task_logger
from flask.globals import current_app

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import PluginState

from . import plugin
from .config import CONFIG_KEY, get_config_from_registry

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{plugin.WorkflowEditor.instance.identifier}.update_config",
    bind=True,
    ignore_result=True,
)
def update_config(self):
    saved_config = get_config_from_registry(current_app)
    if saved_config:
        saved_config["_updated"] = time()
        PluginState.set_value(
            plugin.WorkflowEditor.instance.name, CONFIG_KEY, saved_config, commit=True
        )
    else:
        PluginState.delete_value(plugin.WorkflowEditor.instance.name, CONFIG_KEY)
