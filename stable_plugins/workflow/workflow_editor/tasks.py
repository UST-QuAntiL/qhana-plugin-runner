from collections import defaultdict
from time import time
from typing import Literal, Optional

from celery.utils.log import get_task_logger
from flask.globals import current_app
from requests import request
from werkzeug.utils import secure_filename

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from qhana_plugin_runner.registry_client import PLUGIN_REGISTRY_CLIENT

from . import plugin
from .config import CONFIG_KEY, WF_STATE_KEY, get_config_from_registry
from .parser import get_ad_hoc_tree, split_ui_template_workflow, tree_to_template_tabs
from .util import extract_wf_properties

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
def deploy_workflow(
    self,
    workflow_url: str,
    workflow_id,
    deploy_as: Literal["plugin", "workflow", "ui-template"] = "plugin",
):
    if deploy_as == "plugin":
        with PLUGIN_REGISTRY_CLIENT as client:
            deploy_workflow_plugin = client.search_by_rel(
                "plugin", {"name": "deploy-workflow"}, allow_collection_resource=False
            )
            if not deploy_workflow_plugin:
                return  # TODO: log an error??

        deployment_url = deploy_workflow_plugin.data["entryPoint"]["href"]
        request("post", deployment_url, data={"workflow": workflow_url}, timeout=30)
    elif deploy_as == "workflow":
        plugin_instance = plugin.WorkflowEditor.instance
        if not plugin_instance:
            return  # TODO: log an error??

        config = plugin_instance.get_config()

        camunda_endpoint = config["CAMUNDA_ENDPOINT"]
        worker_id = config["camunda_worker_id"]

        bpmn = DataBlob.get_value(plugin_instance.name, workflow_id, default=None)
        if not bpmn:
            return  # TODO: log an error??

        id_, name, _ = extract_wf_properties(bpmn=bpmn)

        request(
            "post",
            url=f"{camunda_endpoint}/deployment/create",
            params={
                "deployment-name": id_,
                "enable-duplicate-filtering": "true",
                "deployment-source": worker_id,
            },
            files={id_: (secure_filename(name + ".bpmn"), bpmn, "application/xml")},
            timeout=30,
        )
    elif deploy_as == "ui-template":
        _deploy_as_ui_template(workflow_id)


def _deploy_as_ui_template(workflow_id):
    plugin_instance = plugin.WorkflowEditor.instance
    if not plugin_instance:
        return  # TODO: log an error??

    bpmn = DataBlob.get_value(plugin_instance.name, workflow_id, default=None)
    if not bpmn:
        return  # TODO: log an error??

    bpmn, *child_workflows = split_ui_template_workflow(bpmn)

    if child_workflows:
        # TODO implement!
        raise NotImplementedError(
            "Deploying extracted worflows as plugins is not implemented yet."
        )

    _, name, _ = extract_wf_properties(bpmn=bpmn)
    ad_hoc_tree = get_ad_hoc_tree()

    template_data = {
        "name": name,
        "description": f"UI Template generated from the '{name}' workflow.",
        "tags": ["workflow", "generated"],
    }
    ui_template_tabs = tree_to_template_tabs(ad_hoc_tree)

    with PLUGIN_REGISTRY_CLIENT as client:
        response = client.search_by_rel("ui-template")
        if not response:
            return  # TODO log error

        create_links = response.get_links_by_rel("create", "ui-template")
        create_ui_template_link = create_links[0] if create_links else None
        if not create_ui_template_link:
            return  # TODO log error

        template_response = client.fetch_by_api_link(
            create_ui_template_link, json=template_data
        )
        create_links = template_response.get_links_by_rel("create", "ui-template-tab")
        create_tab_link = create_links[0] if create_links else None
        if not create_tab_link:
            return  # TODO log error

        for tab in ui_template_tabs:
            client.fetch_by_api_link(create_tab_link, json=tab)


@CELERY.task()
def cleanup_autosaved_workflows(plugin_name: str):
    """
    Removes autosaved workflows, so that only the latest three are kept for each workflow.

    Args:
        plugin_name:

    Returns:

    """
    saved_workflows = PluginState.get_value(plugin_name, WF_STATE_KEY)
    autosave_count = defaultdict(int)
    cleaned_list = []

    for wf in saved_workflows:
        if wf.get("autosave") is False:
            cleaned_list.append(wf)
            continue

        wf_name = wf.get("id")
        wf_id = wf.get("workflow_id")

        if autosave_count[wf_name] >= 3:
            TASK_LOGGER.info(f"Deleting autosaved workflow of {wf_name} with id {wf_id}")
            DataBlob.delete_value(plugin_name, wf_id)
        else:
            autosave_count[wf_name] += 1
            cleaned_list.append(wf)

    PluginState.set_value(plugin_name, WF_STATE_KEY, cleaned_list, commit=True)
