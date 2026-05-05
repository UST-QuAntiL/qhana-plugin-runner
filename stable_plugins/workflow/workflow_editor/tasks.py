from collections import defaultdict
from time import time
from typing import Dict, Literal, Optional
from xml.etree import ElementTree

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
from .splitting import FragmentResult

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

        id_, name, _ = extract_wf_properties(bpmn=bpmn.decode("utf-8"))

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
        TASK_LOGGER.error("No WorkflowEditor plugin instance available.")
        return

    bpmn_bytes = DataBlob.get_value(plugin_instance.name, workflow_id, default=None)
    if not bpmn_bytes:
        TASK_LOGGER.error(f"No BPMN data found for workflow_id={workflow_id!r}.")
        return

    main_xml, fragments = split_ui_template_workflow(bpmn_bytes.decode(encoding="utf-8"))

    topic_to_plugin_id: Dict[str, str] = {}

    if fragments:
        original_pid = _get_main_original_pid(main_xml)

        for frag in fragments:
            deployed_plugin_id = _deploy_fragment_as_plugin(frag)
            if deployed_plugin_id:
                topic = f"plugin-step.{original_pid}-{frag.fragment_id}"
                topic_to_plugin_id[topic] = deployed_plugin_id

    if topic_to_plugin_id:
        main_xml = _patch_main_xml(main_xml, topic_to_plugin_id)

    _, name, _ = extract_wf_properties(bpmn=main_xml)
    ad_hoc_tree = get_ad_hoc_tree(bpmn=main_xml)

    template_data = {
        "name": name,
        "description": f"UI Template generated from the '{name}' workflow.",
        "tags": ["workflow", "generated"],
    }
    ui_template_tabs = tree_to_template_tabs(ad_hoc_tree)

    with PLUGIN_REGISTRY_CLIENT as client:
        response = client.search_by_rel("ui-template")
        if not response:
            TASK_LOGGER.error("Could not find ui-template API endpoint.")
            return

        create_links = response.get_links_by_rel("create", "ui-template")
        create_ui_template_link = create_links[0] if create_links else None
        if not create_ui_template_link:
            TASK_LOGGER.error("No 'create' link found for ui-template.")
            return

        created_response = client.fetch_by_api_link(
            create_ui_template_link, json=template_data
        )
        assert created_response
        template_response = client.fetch_by_api_link(created_response.data["new"])
        assert template_response
        create_links = template_response.get_links_by_rel("create", "ui-template-tab")
        create_tab_link = create_links[0] if create_links else None
        if not create_tab_link:
            TASK_LOGGER.error("No 'create' link found for ui-template-tab.")
            return

        for tab in ui_template_tabs:
            r = client.fetch_by_api_link(create_tab_link, json=tab)
            assert r


def _get_main_original_pid(main_xml: str) -> str:
    root = ElementTree.fromstring(main_xml)
    for proc in root.iter(f"{{http://www.omg.org/spec/BPMN/20100524/MODEL}}process"):
        pid = proc.get("id") or ""
        if pid.endswith("_main"):
            return pid[: -len("_main")]
        return pid
    return "workflow"


CAMUNDA_NS_URI = "http://camunda.org/schema/1.0/bpmn"
BPMN_MODEL_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"


def _deploy_fragment_as_plugin(
    frag: FragmentResult,
) -> Optional[str]:
    plugin_instance = plugin.WorkflowEditor.instance
    if not plugin_instance:
        TASK_LOGGER.error("No WorkflowEditor plugin instance available.")
        return None

    frag_blob_id = f"fragment_{frag.fragment_id}_{frag.process_id}"
    DataBlob.set_value(
        plugin_instance.name,
        frag_blob_id,
        frag.xml.encode("utf-8"),
        commit=True,
    )

    plugin_runner_url = current_app.config.get(
        "PLUGIN_RUNNER_URLS", "http://localhost:5005"
    )
    if isinstance(plugin_runner_url, list):
        plugin_runner_url = plugin_runner_url[0]
    plugin_runner_url = plugin_runner_url.rstrip("/")
    workflow_url = f"{plugin_runner_url}/plugins/{plugin_instance.identifier}/workflows/{frag_blob_id}/"

    with PLUGIN_REGISTRY_CLIENT as client:
        deploy_workflow_plugin = client.search_by_rel(
            "plugin", {"name": "deploy-workflow"}, allow_collection_resource=False
        )
        if not deploy_workflow_plugin:
            TASK_LOGGER.error("deploy-workflow plugin not found in registry.")
            return None

    deployment_url = deploy_workflow_plugin.data["entryPoint"]["href"]

    try:
        resp = request(
            "post", deployment_url, data={"workflow": workflow_url}, timeout=30
        )
        resp.raise_for_status()
    except Exception:
        TASK_LOGGER.exception(
            f"Failed to deploy fragment {frag.fragment_id} via deploy-workflow plugin."
        )
        return None

    TASK_LOGGER.info(f"Deployed fragment {frag.fragment_id} via deploy-workflow plugin.")

    return frag.process_id


def _patch_main_xml(
    main_xml: str,
    topic_to_plugin_id: Dict[str, str],
) -> str:
    root = ElementTree.fromstring(main_xml)

    for service_task in root.iter(f"{{{BPMN_MODEL_NS}}}serviceTask"):
        topic = service_task.get(f"{{{CAMUNDA_NS_URI}}}topic") or ""
        if topic not in topic_to_plugin_id:
            continue

        deployed_id = topic_to_plugin_id[topic]

        service_task.set(f"{{{CAMUNDA_NS_URI}}}topic", f"plugin.{deployed_id}")

        ext = service_task.find(f"{{{BPMN_MODEL_NS}}}extensionElements")
        if ext is None:
            ext = ElementTree.SubElement(
                service_task, f"{{{BPMN_MODEL_NS}}}extensionElements"
            )

        io = ext.find(f"{{{CAMUNDA_NS_URI}}}inputOutput")
        if io is None:
            io = ElementTree.SubElement(ext, f"{{{CAMUNDA_NS_URI}}}inputOutput")

        id_param = ElementTree.SubElement(
            io, f"{{{CAMUNDA_NS_URI}}}inputParameter", attrib={"name": "qhanaIdentifier"}
        )
        id_param.text = deployed_id

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True).decode(
        "utf-8"
    )


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
