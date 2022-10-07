import requests
from celery.utils.functional import maybe_list
from celery.utils.log import get_task_logger
from requests.exceptions import ConnectionError, HTTPError

from qhana_plugin_runner.celery import CELERY

from ... import Workflows
from ...exceptions import (
    BadTaskDefinitionError,
    BadInputsError,
    InvocationError,
    PluginFailureError,
    PluginNotFoundError,
    ResultError,
)
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


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.qhana_instance_watcher",
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=10,
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
        raise BadTaskDefinitionError(
            message=f"Malformed task topic name '{external_task.topic_name}'. Name must contain a '.' separating the plugin name!"
        )

    plugin = qhana_client.resolve(plugin_name)
    if plugin is None:
        raise PluginNotFoundError(message=f"Plugin {plugin_name} could not be found!")

    workflow_local_variables = camunda_client.get_task_local_variables(external_task)
    try:
        parameters = qhana_client.collect_input(
            external_task,
            camunda_client,
            workflow_local_variables if workflow_local_variables else {},
        )
    except ParameterParsingError as err:
        raise BadTaskDefinitionError(
            message=f"Unsupported input mode '{err.mode}' of input '{err.parameter}'!"
        )

    try:
        status, url = qhana_client.call_qhana_plugin(plugin, parameters)
    except HTTPError as err:
        if err.response and err.response.status_code:
            response_status: int = err.response.status_code
            if response_status == 404:
                raise PluginNotFoundError(
                    message=f"Plugin {plugin_name} endpoint could not be found at the registered url!"
                )
            if 400 <= response_status < 500:
                raise BadInputsError(
                    message="Plugin invocation received unprocessable entities and could not proceed."
                )
            if 500 <= response_status < 600:
                raise InvocationError(
                    message="Plugin invocation failed because of a server error."
                )
        raise InvocationError(
            message="Plugin invocation received unprocessable entities and could not proceed."
        )

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
    autoretry_for=(TaskNotFinishedError,),
    retry_backoff=True,
    max_retries=None,
)
def check_task_status(self, url: str, camunda_external_task: str):
    external_task: ExternalTask = ExternalTask.from_dict(camunda_external_task)

    # TODO: Timeout if no result after a long time (or workflow task removed)
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
        return self.retry(
            countdown=5 * 60,
            max_retries=5,
            exc=ResultError(message="QHAna plugin result endpoint could not be reached."),
        )
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

        raise ResultError(message=error_message)

    # TODO check for substeps!

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

        # Throw bpmn error
        raise PluginFailureError(message="QHAna plugin failed execution.")

    raise TaskNotFinishedError  # retry task
