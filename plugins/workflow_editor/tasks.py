from time import time
from typing import Optional, Literal

from celery.utils.log import get_task_logger
from flask.globals import current_app
from requests import request

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import PluginState
from qhana_plugin_runner.registry_client import PluginRegistryClient

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


@CELERY.task(
    name=f"{plugin.WorkflowEditor.instance.identifier}.deploy_workflow",
    bind=True,
    ignore_result=True,
)
def deploy_workflow(self, workflow_url: str, deploy_as: Literal["plugin", "workflow"] = "plugin"):
    if deploy_as == "plugin":
        with PluginRegistryClient(current_app) as client:
            deploy_workflow_plugin = client.search_by_rel(
                "plugin", {"name": "deploy-workflow"}, allow_collection_resource=False
            )
            if not deploy_workflow_plugin:
                return  # TODO: log an error??

        deployment_url = deploy_workflow_plugin.data["entryPoint"]["href"]
        request("post", deployment_url, data={"workflow": workflow_url}, timeout=30)
    elif deploy_as == "workflow":
        # TODO: deploy workflow to camunda and maybe to QHAna
        # (i.e., using the existing workflow management plugins)
        pass


