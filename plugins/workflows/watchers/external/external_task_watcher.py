from typing import Optional

import requests
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY

from .qhana_instance_watcher import qhana_instance_watcher
from ... import Workflows
from ...clients.camunda_client import CamundaClient
from ...datatypes.camunda_datatypes import CamundaConfig, ExternalTask
from ...exceptions import (
    BadInputsError,
    BadTaskDefinitionError,
    ExecutionError,
    InvocationError,
    PluginNotFoundError,
    ResultError,
    WorkflowTaskError,
)

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.camunda_task_watcher",
    ignore_result=True,
)
def camunda_task_watcher():
    """
    Watches for new Camunda external task. For each new task found a qhana_watcher celery task is spawned.
    """
    # TODO later: rate limit the jobs a single worker takes on at once and don't lock jobs until they are actually started

    # Client
    camunda_client = CamundaClient(
        CamundaConfig(
            base_url=config["CAMUNDA_BASE_URL"],
            poll_interval=config["polling_rates"]["camunda_general"],
        )
    )

    response = requests.get(
        f"{camunda_client.camunda_config.base_url}/external-task",
        timeout=config.get("request_timeout", 5 * 60),
    )
    if response.status_code != 200:
        # no external tasks found => try again next time
        return  # TODO backoff?

    external_tasks = response.json()
    external_tasks = [
        ExternalTask.deserialize(external_task) for external_task in external_tasks
    ]

    TASK_LOGGER.debug(
        f"Searching external task topics with prefix '{camunda_client.camunda_config.plugin_prefix}.'"
    )

    for external_task in external_tasks:
        topic_name = external_task.topic_name

        # Check if task is already locked
        if camunda_client.is_locked(external_task):
            continue

        # Check if the external task represents a qhana plugin
        if topic_name.startswith(f"{camunda_client.camunda_config.plugin_prefix}."):
            # Lock task for usage and to block other watchers from access
            camunda_client.lock(external_task)

            # Serialize
            external_task = external_task.to_dict()
            TASK_LOGGER.debug(f"Start watcher for camunda task {external_task}.")
            # Spawn new watcher for the external task
            instance_task = qhana_instance_watcher.s(external_task)
            # FIXME unlocked tasks should not be started again/moved to a dead letter queue after x tries.../after certain errors...
            instance_task.link_error(
                process_workflow_error.s(camunda_external_task=external_task)
            )
            instance_task.link_error(unlock_task.si(camunda_external_task=external_task))
            instance_task.apply_async()


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.process_workflow_error",
    ignore_result=True,
)
def process_workflow_error(request, exc, traceback, camunda_external_task: dict):
    TASK_LOGGER.error(f"Inputs: exc={exc}, external_task={camunda_external_task}")
    if "hello" in camunda_external_task:
        return
    external_task: ExternalTask = ExternalTask.from_dict(camunda_external_task)

    camunda_client = CamundaClient(
        CamundaConfig(
            base_url=config["CAMUNDA_BASE_URL"],
            poll_interval=config["polling_rates"]["camunda_general"],
        )
    )

    error_code_prefix = config["workflow_error_prefix"]

    error_code = "unknown-error"
    message: Optional[str] = None

    if isinstance(exc, WorkflowTaskError):
        message = exc.message

    if isinstance(exc, BadTaskDefinitionError):
        error_code = "unprocessable-task-definition-error"
    elif isinstance(exc, BadInputsError):
        error_code = "unprocessable-entity-error"
    elif isinstance(exc, PluginNotFoundError):
        error_code = "plugin-not-found-error"
    elif isinstance(exc, InvocationError):
        error_code = "unprocessable-entity-error"
    elif isinstance(exc, ResultError):
        error_code = "unprocessable-plugin-result-error"
    elif isinstance(exc, ExecutionError):
        error_code = "plugin-failure"

    if message is None:
        message = str(exc)

    camunda_client.external_task_bpmn_error(
        task=external_task,
        error_code=f"{error_code_prefix}-{error_code}",
        error_message=message,
    )


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.camunda_unlock_task",
    ignore_result=True,
)
def unlock_task(camunda_external_task: dict):
    external_task: ExternalTask = ExternalTask.from_dict(camunda_external_task)

    camunda_client = CamundaClient(
        CamundaConfig(
            base_url=config["CAMUNDA_BASE_URL"],
            poll_interval=config["polling_rates"]["camunda_general"],
        )
    )
    try:
        camunda_client.unlock(external_task)
    except:
        # FIXME fix exception handling in client function (should throw more specific exception)
        pass
