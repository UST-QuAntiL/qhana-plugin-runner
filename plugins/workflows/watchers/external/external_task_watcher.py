from typing import Optional

from celery.utils.log import get_task_logger
from requests.exceptions import RequestException

from qhana_plugin_runner.celery import CELERY

from .qhana_instance_watcher import qhana_instance_watcher, qhana_step_watcher
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


def execute_task(external_task: ExternalTask, camunda_client: CamundaClient, watcher):
    if external_task.execution_id is None:
        raise BadTaskDefinitionError(
            message=f"External task {external_task} has no execution id!"
        )
    # Lock task for usage and to block other watchers from access
    camunda_client.lock(external_task.id)

    TASK_LOGGER.debug(f"Start watcher for camunda task {external_task}.")
    # Spawn new watcher for the external task
    instance_task = watcher.s(
        topic_name=external_task.topic_name,
        external_task_id=external_task.id,
        execution_id=external_task.execution_id,
        process_instance_id=external_task.process_instance_id,
    )
    instance_task.link_error(process_workflow_error.s(external_task_id=external_task.id))
    instance_task.link_error(unlock_task.si(external_task_id=external_task.id))
    instance_task.apply_async()


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.camunda_task_watcher",
    ignore_result=True,
)
def camunda_task_watcher():
    """
    Watches for new Camunda external task. For each new task found a qhana_watcher celery task is spawned.
    """
    # Client
    camunda_client = CamundaClient(CamundaConfig.from_config(config))
    max_concurrent_external_tasks = config["external_task_concurrency"]

    TASK_LOGGER.info(
        f"Polling for external tasks as worker '{camunda_client.camunda_config.worker_id}'."
    )

    try:
        locked_task_count = camunda_client.get_locked_external_tasks_count()
        max_tasks = max_concurrent_external_tasks - locked_task_count
        if max_tasks < 1:
            TASK_LOGGER.debug("Reached external task concurrency limit!")
            return  # only lock x tasks in parallel, retry until some tasks are finished
        TASK_LOGGER.debug(
            f"Waiting on {locked_task_count} external tasks, fetching <= {max_tasks} new tasks."
        )
        external_tasks = camunda_client.get_external_tasks(limit=max_tasks)
        TASK_LOGGER.info(f"Received {len(external_tasks)} tasks.")
    except RequestException as err:
        TASK_LOGGER.info(f"Error retrieving external tasks from camunda: {err}")
        return

    TASK_LOGGER.debug(
        f"Searching external task topics with prefix '{camunda_client.camunda_config.plugin_prefix}.'"
    )

    for external_task in external_tasks:
        topic_name = external_task.topic_name

        # Check if task is already locked
        if camunda_client.is_locked(external_task.id):
            continue

        if topic_name.startswith(f"{camunda_client.camunda_config.plugin_step_prefix}."):
            # task is a qhana step
            execute_task(
                external_task=external_task,
                camunda_client=camunda_client,
                watcher=qhana_step_watcher,
            )
        elif topic_name.startswith(f"{camunda_client.camunda_config.plugin_prefix}."):
            # task is a qhana plugin
            execute_task(
                external_task=external_task,
                camunda_client=camunda_client,
                watcher=qhana_instance_watcher,
            )


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.process_workflow_error",
    ignore_result=True,
)
def process_workflow_error(request, exc, traceback, external_task_id: str):
    camunda_client = CamundaClient(CamundaConfig.from_config(config=config))

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
        external_task_id=external_task_id,
        error_code=f"{error_code_prefix}-{error_code}",
        error_message=message,
    )


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.camunda_unlock_task",
    ignore_result=True,
)
def unlock_task(external_task_id: str):
    camunda_client = CamundaClient(CamundaConfig.from_config(config=config))
    try:
        camunda_client.unlock(external_task_id)
    except:
        # FIXME fix exception handling in client function (should throw more specific exception)
        pass
