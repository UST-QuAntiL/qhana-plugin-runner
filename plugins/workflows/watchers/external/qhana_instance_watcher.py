import requests
from celery.utils.functional import maybe_list
from celery.utils.log import get_task_logger
from requests.exceptions import ConnectionError, HTTPError

from qhana_plugin_runner.celery import CELERY

from ... import Workflows
from ...clients.camunda_client import CamundaClient
from ...clients.qhana_task_client import ParameterParsingError, QhanaTaskClient
from ...datatypes.camunda_datatypes import CamundaConfig, ExternalTask
from ...datatypes.qhana_datatypes import QhanaOutput, QhanaTask

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


class TaskNotFinishedError(Exception):
    pass


def get_camunda_client():
    return CamundaClient(
        CamundaConfig(
            base_url=config["CAMUNDA_BASE_URL"],
            poll_interval=config["polling_rates"]["camunda_general"],
        )
    )


def extract_plugin_name(topic: str):
    split_topic = topic.split(".", maxsplit=1)
    if len(split_topic) <= 1:
        raise ValueError("Topic must contain a '.' to separate the plugin name.")
    return split_topic[1]


def finish_task_with_error(
    external_task: ExternalTask, error_code: str, error_message: str
):
    camunda_client = get_camunda_client()

    # Throw bpmn error
    camunda_client.external_task_bpmn_error(
        task=external_task,
        error_code=error_code,
        error_message=error_message,
    )


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.qhana_instance_watcher",
    bind=True,
    ignore_result=True,
)
def qhana_instance_watcher(self, camunda_external_task: str):
    """
    Creates new qhana plugin instances and watches for results.
    """
    TASK_LOGGER.info(f"Received task from camunda queue: {camunda_external_task}")

    # Clients
    camunda_client = get_camunda_client()
    TASK_LOGGER.debug(f"Searching for plugins")
    qhana_client = QhanaTaskClient(config["QHANA_PLUGIN_ENDPOINTS"])

    # Deserialize
    external_task: ExternalTask = ExternalTask.from_dict(camunda_external_task)
    TASK_LOGGER.debug(f"Invoke plugin")

    try:
        plugin_name = extract_plugin_name(external_task.topic_name)
    except ValueError:
        finish_task_with_error(
            external_task=external_task,
            error_code="qhana-unprocessable-entity-error",
            error_message=f"Malformed task topic name '{external_task.topic_name}'. Name must contain a '.' separating the plugin name!",
        )
        return

    plugin = qhana_client.resolve(plugin_name)
    if plugin is None:
        finish_task_with_error(
            external_task=external_task,
            error_code="qhana-plugin-not-found-error",
            error_message=f"Plugin {plugin_name} could not be found!",
        )
        return

    workflow_local_variables = camunda_client.get_task_local_variables(external_task)
    try:
        parameters = qhana_client.collect_input(
            external_task,
            camunda_client,
            workflow_local_variables if workflow_local_variables else {},
        )
    except ParameterParsingError as err:
        finish_task_with_error(
            external_task=external_task,
            error_code="qhana-mode-error",
            error_message=f"Unsupported input mode '{err.mode}' of input '{err.parameter}'!",
        )
        return

    try:
        status, url = qhana_client.call_qhana_plugin(plugin, parameters)
    except HTTPError:
        # TODO retries on timeouts and server errors
        finish_task_with_error(
            external_task=external_task,
            error_code="qhana-unprocessable-entity-error",
            error_message="Plugin invocation received unprocessable entities and could not proceed.",
        )
        return

    qhana_instance = QhanaTask(
        status=status,
        url=url,
        external_task=external_task,
        plugin=plugin,
    )

    TASK_LOGGER.info(
        f"Created QHAna plugin instance {qhana_instance.plugin.name} with result url: {qhana_instance.url}"
    )

    watch_task = check_task_status.s(
        url=qhana_instance.url,
        camunda_external_task=camunda_external_task,
    )
    errbacks = maybe_list(self.request.errbacks)
    if errbacks:
        for error_handler in errbacks:
            watch_task.link_error(error_handler)
    watch_task.apply_async()


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.check_task_status",
    bind=True,
    ignore_result=True,
    autoretry_for=(TaskNotFinishedError,),
    retry_backoff=True,
    max_retries=None,
)
def check_task_status(self, url: str, camunda_external_task: str):
    external_task: ExternalTask = ExternalTask.from_dict(camunda_external_task)

    # TODO: Timeout if no result after a long time
    try:
        response = requests.get(url, timeout=config.get("request_timeout", 5 * 60))
        response.raise_for_status()
        contents = response.json()

        qhana_instance_status = contents["status"]
    except ConnectionError:
        TASK_LOGGER.warning(
            f"QHAna plugin result under '{url}' could not be reached. Throwing BPMN exception...",
            exc_info=True,
        )

        finish_task_with_error(
            external_task=external_task,
            error_code="qhana-plugin-unreachable",
            error_message="QHAna plugin result endpoint could not be reached.",
        )
        return
    except HTTPError as err:
        status: int = err.response.status_code
        if 500 <= status < 600:
            # is server error, retry may help
            raise TaskNotFinishedError  # retry task
        error_message: str = (
            f"QHAna plugin result endpoint did respond with an http error {status}."
        )
        if status == 404:
            error_message = "QHAna plugin result was not found."

        finish_task_with_error(
            external_task=external_task,
            error_code="qhana-plugin-failure",
            error_message=error_message,
        )
        return

    # Check if qhana task completed successfully
    if qhana_instance_status == "SUCCESS":
        TASK_LOGGER.info(f"QHAna plugin completed successfully. Result Resource: {url}")

        outputs = [QhanaOutput.deserialize(output) for output in contents["outputs"]]

        qhana_client = QhanaTaskClient(config["QHANA_PLUGIN_ENDPOINTS"])
        camunda_client = get_camunda_client()

        # Complete external task with qhana task result
        qhana_client.complete_qhana_task(
            camunda_client, outputs=outputs, external_task=external_task
        )

        return
    elif qhana_instance_status == "FAILURE":
        TASK_LOGGER.info(
            f"QHAna plugin failed, throwing BPMN exception. Result Resource: {url}"
        )

        camunda_client = get_camunda_client()

        # Throw bpmn error
        camunda_client.external_task_bpmn_error(
            task=external_task,
            error_code="qhana-plugin-failure",
            error_message="QHAna plugin failed execution.",
        )

        return

    raise TaskNotFinishedError  # retry task
