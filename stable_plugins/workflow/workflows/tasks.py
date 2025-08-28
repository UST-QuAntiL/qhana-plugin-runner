import json
from typing import Mapping, Optional

from celery.utils.log import get_task_logger
from flask import current_app
from requests.exceptions import ConnectionError, HTTPError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.virtual_plugins import (
    VIRTUAL_PLUGIN_CREATED,
    PluginState,
    VirtualPlugin,
)
from qhana_plugin_runner.tasks import TASK_STEPS_CHANGED

from . import DeployWorkflow
from .clients.camunda_client import CamundaClient, CamundaManagementClient
from .exceptions import CamundaClientError, CamundaServerError, WorkflowNotFoundError

config = DeployWorkflow.instance.config

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{DeployWorkflow.instance.identifier}.deploy_virtual_plugin")
def deploy_virtual_plugin(plugin_url: str, process_definition_id: str):
    from .management import WorkflowManagement

    if VirtualPlugin.exists([VirtualPlugin.href == plugin_url]):
        return plugin_url

    camunda = CamundaManagementClient(config)
    try:
        process_instance = camunda.get_process_definition(
            definition_id=process_definition_id
        )
    except HTTPError as err:
        if err.response is not None and err.response.status_code == 404:
            raise WorkflowNotFoundError(
                message="Process definition does not exist."
            ) from err
        raise

    version: str = str(process_instance.get("version", 1))
    description: Optional[str] = process_instance.get("description")
    key: Optional[str] = process_instance.get("key")

    plugin = VirtualPlugin(
        parent_id=WorkflowManagement.instance.identifier,
        name=key if key else process_definition_id,
        version=version,
        tags="\n".join(["workflow", "bpmn"]),
        description=description if description else "",
        href=plugin_url,
    )

    variables = camunda.get_workflow_start_form_variables(process_definition_id)
    if variables:
        PluginState.set_value(
            WorkflowManagement.instance.identifier, plugin_url, variables
        )

    DB.session.add(plugin)
    DB.session.commit()

    VIRTUAL_PLUGIN_CREATED.send(current_app._get_current_object(), plugin_url=plugin_url)

    return plugin_url


@CELERY.task(name=f"{DeployWorkflow.instance.identifier}.deploy_workflow", bind=True)
def deploy_workflow(self, db_id: int) -> None:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(f"Could not load task data for task id '{db_id}'!")
        raise ValueError(f"Task {db_id} was not found!")

    workflow_file_url = task_data.parameters

    camunda = CamundaManagementClient(config)

    TASK_LOGGER.info(f"Deploying workflow using BPMN file: '{workflow_file_url}'")
    process_definition_id = camunda.deploy_bpmn(bpmn_url=workflow_file_url)
    TASK_LOGGER.info(f"Workflow deployed as: '{process_definition_id}'")

    assert isinstance(task_data.data, dict)

    plugin_url_template = task_data.data["plugin_url_template"]
    assert isinstance(plugin_url_template, str)

    task_data.add_task_log_entry(
        f"Deployed workflow with id '{process_definition_id}'.", commit=True
    )

    plugin_url = plugin_url_template.replace(
        "%7Bprocess_definition_id%7D", process_definition_id
    )

    self.replace(
        deploy_virtual_plugin.si(
            plugin_url=plugin_url, process_definition_id=process_definition_id
        )
    )


@CELERY.task(
    name=f"{DeployWorkflow.instance.identifier}.start_workflow_with_arguments",
    bind=True,
    autoretry_for=(ConnectionError, CamundaServerError),
    retry_backoff=True,
    max_retries=10,
)
def start_workflow_with_arguments(self, db_id: int, workflow_id: str) -> None:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(f"Could not load task data for task id '{db_id}'!")
        raise ValueError(f"Task {db_id} was not found!")

    workflow_inputs: Mapping = json.loads(
        task_data.parameters if task_data.parameters else "{}"
    )

    camunda = CamundaManagementClient(config)

    try:
        process_instance = camunda.start_workflow(
            definition_id=workflow_id, form_inputs=workflow_inputs
        )
    except HTTPError as err:
        print(err.request.headers)
        if err.response is not None and err.response.status_code:
            response_status: int = err.response.status_code
            if response_status == 404:
                raise WorkflowNotFoundError(
                    message=f"Workflow {workflow_id} could not be found!"
                ) from err
            if 400 <= response_status < 500:
                TASK_LOGGER.warning(
                    "Starting the workflow failed because of a client error. (Message body: {{}})",
                    err.response.text,
                    exc_info=False,
                )
                raise CamundaClientError(
                    message="Workflow start received unprocessable entities and could not proceed."
                ) from err
            if 500 <= response_status < 600:
                raise CamundaServerError(
                    message="Starting the workflow failed because of a server error."
                ) from err
        TASK_LOGGER.warning(
            f"Starting the workflow failed because of an unknown error. (Message body: {err.response.text})",
            exc_info=False,
        )
        raise
    TASK_LOGGER.info(f"Started process instance with id '{process_instance['id']}'!")

    assert isinstance(task_data.data, dict)

    # Set the process instance id used to create Camunda configs in sub-steps
    task_data.data["camunda_process_instance_id"] = process_instance["id"]
    task_data.data["camunda_process_definition_id"] = workflow_id
    task_data.add_task_log_entry("Started workflow.")
    task_data.save(commit=True)


@CELERY.task(name=f"{DeployWorkflow.instance.identifier}.process_input", bind=True)
def process_input(self, db_id: int) -> None:
    TASK_LOGGER.info(f"Started input processing with db_id: '{db_id}'")

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(f"Could not load task data for task id '{db_id}'!")
        raise ValueError(f"Task {db_id} was not found!")

    assert isinstance(task_data.data, dict)

    input_params: dict = json.loads(task_data.parameters)
    human_task_id = task_data.data["human_task_id"]
    assert isinstance(human_task_id, str)

    # Client
    camunda_client = CamundaClient(config)

    variables = {
        key: {
            "value": val,
        }
        for key, val in input_params.items()
    }

    status = camunda_client.get_human_task_status(human_task_id)

    if status in ("PENDING", None):
        # only attempt to complete tasks that are still pending/untaken
        camunda_client.complete_human_task(human_task_id, variables)
    if status == "ERROR":
        task_data.add_task_log_entry(
            f"Task with ID '{human_task_id}' could not be completed because of an unknown error!"
        )

    task_data.data.pop("external_form_key", None)

    task_data.clear_previous_step()
    task_data.save(commit=True)

    app = current_app._get_current_object()
    TASK_STEPS_CHANGED.send(app, task_id=db_id)
