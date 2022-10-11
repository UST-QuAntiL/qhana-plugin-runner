import requests
from celery.utils.functional import maybe_list
from celery.utils.log import get_task_logger
from requests.exceptions import ConnectionError, HTTPError

from qhana_plugin_runner.celery import CELERY

from ... import Workflows
from ...clients.camunda_client import CamundaClient
from ...clients.qhana_task_client import ParameterParsingError, QhanaTaskClient
from ...datatypes.camunda_datatypes import CamundaConfig
from ...datatypes.qhana_datatypes import QhanaOutput
from ...exceptions import (
    BadInputsError,
    BadTaskDefinitionError,
    InvocationError,
    PluginFailureError,
    PluginNotFoundError,
    ResultError,
    StepNotFoundError,
)

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


class TaskNotFinishedError(Exception):
    pass


def extract_second_topic_component(topic: str):
    split_topic = topic.split(".", maxsplit=1)
    if len(split_topic) <= 1:
        raise ValueError(
            "Topic must contain a '.' to separate the second topic component."
        )
    return split_topic[1]


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.qhana_instance_watcher",
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=10,
)
def qhana_instance_watcher(
    self,
    topic_name: str,
    external_task_id: str,
    execution_id: str,
    process_instance_id: str,
):
    """
    Creates new qhana plugin instances and watches for results.
    """
    TASK_LOGGER.info(f"Received task {external_task_id} from camunda queue {topic_name}.")

    # Clients
    camunda_client = CamundaClient(CamundaConfig.from_config(config))
    TASK_LOGGER.debug(f"Searching for plugins")
    qhana_client = QhanaTaskClient(config["QHANA_PLUGIN_ENDPOINTS"])

    try:
        plugin_name = extract_second_topic_component(topic_name)
    except ValueError:
        raise BadTaskDefinitionError(
            message=f"Malformed task topic name '{topic_name}'. Name must contain a '.' separating the plugin name!"
        )

    plugin = qhana_client.resolve(plugin_name)
    if plugin is None:
        raise PluginNotFoundError(message=f"Plugin {plugin_name} could not be found!")

    workflow_local_variables = camunda_client.get_task_local_variables(execution_id)
    try:
        parameters = qhana_client.collect_input(
            workflow_local_variables if workflow_local_variables else {},
            process_instance_id=process_instance_id,
            camunda_client=camunda_client,
        )
    except ParameterParsingError as err:
        raise BadTaskDefinitionError(
            message=f"Unsupported input mode '{err.mode}' of input '{err.parameter}'!"
        )

    try:
        url = qhana_client.call_qhana_plugin(plugin, parameters)
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

    TASK_LOGGER.info(
        f"Created QHAna plugin instance {plugin.name} with result url: {url}"
    )

    watch_task = check_task_status.s(
        url=url,
        external_task_id=external_task_id,
    )
    errbacks = maybe_list(self.request.errbacks)
    if errbacks:
        for error_handler in errbacks:
            watch_task.link_error(error_handler)
    watch_task.apply_async()


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.qhana_step_watcher",
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    max_retries=10,
)
def qhana_step_watcher(
    self,
    topic_name: str,
    external_task_id: str,
    execution_id: str,
    process_instance_id: str,
):
    """
    Sends inputs to an open plugin step and watches for new results.
    """
    TASK_LOGGER.info(f"Received task {external_task_id} from camunda queue {topic_name}.")

    # Clients
    camunda_client = CamundaClient(CamundaConfig.from_config(config))
    TASK_LOGGER.debug(f"Receiving plugin step.")

    try:
        step_var = extract_second_topic_component(topic_name)
    except ValueError:
        raise BadTaskDefinitionError(
            message=f"Malformed task topic name '{topic_name}'. Name must contain a '.' separating the step variable name!"
        )

    workflow_local_variables = camunda_client.get_task_local_variables(execution_id)

    if step_var not in workflow_local_variables:
        raise BadTaskDefinitionError(message=f"Missing task step variable '{step_var}'.")

    step: dict = workflow_local_variables[step_var]["value"]
    if step.keys() < {"resultUrl", "stepNr", "href"}:
        raise BadTaskDefinitionError(
            message=f"The plugin step to execute is incomplete! Step: {step}"
        )

    qhana_client = QhanaTaskClient(config["QHANA_PLUGIN_ENDPOINTS"])
    try:
        parameters = qhana_client.collect_input(
            workflow_local_variables if workflow_local_variables else {},
            process_instance_id=process_instance_id,
            camunda_client=camunda_client,
        )
    except ParameterParsingError as err:
        raise BadTaskDefinitionError(
            message=f"Unsupported input mode '{err.mode}' of input '{err.parameter}'!"
        )

    step_id = step.get("stepId", step["stepNr"])
    step_href: str = step["href"]
    try:
        qhana_client.call_plugin_step(step_href, parameters)
    except HTTPError as err:
        if err.response and err.response.status_code:
            response_status: int = err.response.status_code
            if response_status == 404:
                raise StepNotFoundError(
                    message=f"Step endpoint '{step_href}' was not be found!"
                )
            if 400 <= response_status < 500:
                raise BadInputsError(
                    message="Step invocation received unprocessable entities and could not proceed."
                )
            if 500 <= response_status < 600:
                raise InvocationError(
                    message="Step invocation failed because of a server error."
                )
        raise InvocationError(
            message="Step invocation received unprocessable entities and could not proceed."
        )

    url = step["resultUrl"]
    TASK_LOGGER.info(f"Sent input to QHAna plugin step {step_id} with result url: {url}")

    watch_task = check_task_status.s(
        url=url,
        external_task_id=external_task_id,
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
def check_task_status(self, url: str, external_task_id: str, last_step_count=0):
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

    # Check if qhana task completed successfully
    if qhana_instance_status == "SUCCESS":
        TASK_LOGGER.info(f"QHAna plugin completed successfully. Result Resource: {url}")

        outputs = [QhanaOutput.deserialize(output) for output in contents["outputs"]]

        camunda_client = CamundaClient(CamundaConfig.from_config(config))

        # Complete external task with qhana task result
        external_task_result = {
            "output": {
                "value": [
                    {
                        "name": output.name,
                        "contentType": output.content_type,
                        "dataType": output.data_type,
                        "href": output.href,
                    }
                    for output in outputs
                ]
            }
        }
        camunda_client.complete_task(external_task_id, external_task_result)
        return  # exit without retry

    elif qhana_instance_status == "FAILURE":
        TASK_LOGGER.info(
            f"QHAna plugin failed, throwing BPMN exception. Result Resource: {url}"
        )

        # Throw bpmn error
        raise PluginFailureError(message="QHAna plugin failed execution.")

    # complete external task on detecting a new step
    steps = contents.get("steps", [])
    if len(steps) > min(last_step_count, 0):
        # found a potentially new step
        next_step = steps[-1]
        if next_step and not next_step.get("cleared", True):
            # True as default because of negation
            # next step is uncleared, finish with next step as the result
            external_task_result = {
                "output": {
                    "value": {
                        "resultUrl": url,
                        "stepNr": len(steps),
                        "stepId": next_step["stepId"],
                        "href": next_step["href"],
                        "uiHref": next_step["uiHref"],
                        # TODO add links to step output
                    }
                }
            }

            camunda_client = CamundaClient(CamundaConfig.from_config(config))
            camunda_client.complete_task(external_task_id, external_task_result)
            return  # exit without retry

    raise TaskNotFinishedError  # retry task
