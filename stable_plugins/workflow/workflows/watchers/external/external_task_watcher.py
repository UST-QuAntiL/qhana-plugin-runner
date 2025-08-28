from typing import Optional, cast

from celery.utils.log import get_task_logger
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    InvalidJSONError,
    RequestException,
)

from qhana_plugin_runner.celery import CELERY

from ... import DeployWorkflow
from ...clients.camunda_client import CamundaClient
from ...config import WorkflowConfig
from ...datatypes.camunda_datatypes import ExternalTask
from ...exceptions import (
    BadInputsError,
    BadTaskDefinitionError,
    CamundaServerError,
    ExecutionError,
    InvocationError,
    PluginNotFoundError,
    ResultError,
    WorkflowTaskError,
)
from .qhana_instance_watcher import (
    QHAnaStepData,
    qhana_instance_watcher,
    qhana_step_watcher,
)

config = DeployWorkflow.instance.config

TASK_LOGGER = get_task_logger(__name__)


def execute_task_legacy(
    external_task: ExternalTask, camunda_client: CamundaClient, watcher
):
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

    instance_task.apply_async()


def execute_task(
    external_task: ExternalTask, camunda_client: CamundaClient, config: WorkflowConfig
):
    if external_task.execution_id is None:
        raise BadTaskDefinitionError(
            message=f"External task {external_task} has no execution id!"
        )
    variables = camunda_client.get_task_local_variables(external_task.id)
    qhana_vars = {config["plugin_variable"], config["step_variable"]} - variables.keys()
    if len(qhana_vars) > 1:
        raise BadTaskDefinitionError(
            message=f"External task {external_task} has conflicting QHAna variables {config['plugin_variable']} and {config['step_variable']}!"
        )
    if not qhana_vars:
        raise BadTaskDefinitionError(
            message=f"External task {external_task} has no QHAna variables! Expected either {config['plugin_variable']} or {config['step_variable']}."
        )

    is_plugin_task = config["plugin_variable"] in variables
    qhana_var_value: str | dict | QHAnaStepData

    if is_plugin_task:
        qhana_var_value = variables[config["plugin_variable"]]["value"]
        if not isinstance(qhana_var_value, str):
            raise BadTaskDefinitionError(
                message=f"The {config['plugin_variable']} variable of external task {external_task} has an invalid value of '{qhana_var_value}', expected a string."
            )
    else:
        qhana_var_value = variables[config["step_variable"]]["value"]
        if not isinstance(qhana_var_value, dict):
            raise BadTaskDefinitionError(
                message=f"The {config['step_variable']} variable of external task {external_task} has an invalid value of '{qhana_var_value}', expected a dict."
            )
        if not all(
            isinstance(qhana_var_value.get(key), a)
            for key, a in QHAnaStepData.__annotations__.items()
        ):
            raise BadTaskDefinitionError(
                message=f"The {config['step_variable']} variable of external task {external_task} has an invalid value of '{qhana_var_value}', expected a dict of with the form of {QHAnaStepData.__annotations__}."
            )
        qhana_var_value = cast(QHAnaStepData, qhana_var_value)

    # Lock task for usage and to block other watchers from access
    camunda_client.lock(external_task.id)

    if is_plugin_task:
        instance_task = qhana_instance_watcher.s(
            topic_name=external_task.topic_name,
            external_task_id=external_task.id,
            execution_id=external_task.execution_id,
            process_instance_id=external_task.process_instance_id,
            plugin=qhana_var_value,
        )
    else:
        instance_task = qhana_step_watcher.s(
            topic_name=external_task.topic_name,
            external_task_id=external_task.id,
            execution_id=external_task.execution_id,
            process_instance_id=external_task.process_instance_id,
            step_data=qhana_var_value,
        )

    instance_task.link_error(process_workflow_error.s(external_task_id=external_task.id))

    instance_task.apply_async()


@CELERY.task(
    name=f"{DeployWorkflow.instance.identifier}.external.camunda_task_watcher",
    ignore_result=True,
)
def camunda_task_watcher():
    """
    Watches for new Camunda external task. For each new task found a qhana_watcher celery task is spawned.
    """
    # Client
    camunda_client = CamundaClient(config)
    max_concurrent_external_tasks = config["max_concurrent_tasks"]

    disable_legacy = config["workflow_conf"]["disable_legacy_support"]

    qhana_task_topic = config["workflow_conf"]["qhana_task_topic"]

    TASK_LOGGER.info(f"Polling for external tasks as worker '{config['worker_id']}'.")
    if not disable_legacy:
        TASK_LOGGER.info("Polling with legacy workflow support enabled.")

    try:
        locked_task_count = camunda_client.get_locked_external_tasks_count()
        max_tasks = max_concurrent_external_tasks - locked_task_count
        if max_tasks < 1:
            TASK_LOGGER.debug("Reached external task concurrency limit!")
            return  # only lock x tasks in parallel, retry until some tasks are finished
        TASK_LOGGER.debug(
            f"Waiting on {locked_task_count} external tasks, fetching <= {max_tasks} new tasks."
        )
        if disable_legacy:
            external_tasks = camunda_client.get_external_tasks(
                limit=max_tasks, topic=qhana_task_topic
            )
        else:
            external_tasks = camunda_client.get_external_tasks(limit=max_tasks)
        TASK_LOGGER.info(f"Received {len(external_tasks)} tasks.")
    except RequestException as err:
        TASK_LOGGER.info(f"Error retrieving external tasks from camunda: {err}")
        return

    legacy_task_topic_prefix = config["workflow_conf"]["legacy_plugin_task_topic_prefix"]
    legacy_step_topic_prefix = config["workflow_conf"]["legacy_step_topic_prefix"]

    # FIXME use non legacy settings here!!!

    TASK_LOGGER.debug(
        f"Searching external task topics with prefixes '{legacy_task_topic_prefix}' and '{legacy_step_topic_prefix}'."
    )

    for external_task in external_tasks:
        topic_name = external_task.topic_name

        # Check if task is already locked
        if camunda_client.is_locked(external_task.id):
            continue

        if topic_name == qhana_task_topic:
            execute_task(
                external_task=external_task,
                camunda_client=camunda_client,
                config=config["workflow_conf"],
            )

        if not disable_legacy:
            if topic_name.startswith(f"{legacy_step_topic_prefix}."):
                # task is a qhana step
                execute_task_legacy(
                    external_task=external_task,
                    camunda_client=camunda_client,
                    watcher=qhana_step_watcher,
                )
            elif topic_name.startswith(f"{legacy_task_topic_prefix}."):
                # task is a qhana plugin
                execute_task_legacy(
                    external_task=external_task,
                    camunda_client=camunda_client,
                    watcher=qhana_instance_watcher,
                )


@CELERY.task(
    name=f"{DeployWorkflow.instance.identifier}.external.process_workflow_error",
    ignore_result=True,
)
def process_workflow_error(request, exc, traceback, external_task_id: str):
    error_code_prefix = config["workflow_conf"]["workflow_error_prefix"]

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

    error_task = send_workflow_error.s(
        external_task_id=external_task_id,
        error_code=f"{error_code_prefix}-{error_code}",
        message=message,
    )
    error_task.link_error(
        send_workflow_task_failure.si(
            external_task_id=external_task_id,
            error_code=f"{error_code_prefix}-{error_code}",
            message=message,
        )
    )
    error_task.link_error(unlock_task.si(external_task_id=external_task_id))

    error_task.apply_async()


@CELERY.task(
    name=f"{DeployWorkflow.instance.identifier}.external.send_workflow_error",
    ignore_result=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=10,
)
def send_workflow_error(external_task_id: str, error_code: str, message: str):
    camunda_client = CamundaClient(config)
    try:
        camunda_client.external_task_bpmn_error(
            external_task_id=external_task_id,
            error_code=error_code,
            error_message=message,
        )
    except HTTPError as err:
        if err.response is not None and err.response.status_code:
            response_status = err.response.status_code
            if 400 <= response_status < 500:
                TASK_LOGGER.error(
                    f"Malformed error request, could not set error for workflow task {external_task_id}!"
                )
            if 500 <= response_status < 600:
                try:
                    server_error = err.response.json()
                    if server_error["type"] == "ProcessEngineException" and server_error[
                        "message"
                    ].startswith("ENGINE-13033 "):
                        camunda_client.external_task_report_failure(
                            external_task_id=external_task_id,
                            error_code=error_code,
                            error_message=message,
                        )
                        return  # successfully reported the task failure!
                except InvalidJSONError as json_err:
                    print(json_err)
                    pass
                TASK_LOGGER.warning(
                    f"Server error when setting task error for task {external_task_id}! (Response body: {err.response.text})"
                )
                raise CamundaServerError(
                    message=f"Error while setting the error message for external task {external_task_id}."
                ) from err


@CELERY.task(
    name=f"{DeployWorkflow.instance.identifier}.external.send_workflow_task_failure",
    ignore_result=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=50,
)
def send_workflow_task_failure(external_task_id: str, error_code: str, message: str):
    camunda_client = CamundaClient(config)
    camunda_client.external_task_report_failure(
        external_task_id=external_task_id,
        error_code=error_code,
        error_message=message,
    )


@CELERY.task(
    name=f"{DeployWorkflow.instance.identifier}.external.camunda_unlock_task",
    ignore_result=True,
)
def unlock_task(external_task_id: str):
    camunda_client = CamundaClient(config)
    try:
        camunda_client.unlock(external_task_id)
    except Exception:
        # FIXME fix exception handling in client function (should throw more specific exception)
        pass
