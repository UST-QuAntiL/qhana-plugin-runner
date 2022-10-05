from celery.utils.log import get_task_logger
from celery.utils.functional import maybe_list
from requests.exceptions import ConnectionError

from qhana_plugin_runner.celery import CELERY

from ... import Workflows
from ...clients.camunda_client import CamundaClient
from ...clients.qhana_task_client import QhanaTaskClient
from ...datatypes.camunda_datatypes import CamundaConfig, ExternalTask
from ...datatypes.qhana_datatypes import QhanaOutput
from ...util.helper import request_json

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

    # Create new qhana plugin instance
    qhana_instance = qhana_client.create_qhana_plugin_instance(
        camunda_client, external_task
    )

    TASK_LOGGER.info(
        f"Created QHAna plugin instance {qhana_instance.plugin.name} with id: {qhana_instance.id}"
    )

    watch_task = check_task_status.s(
        url=f"{qhana_instance.plugin.api_endpoint}/tasks/{qhana_instance.id}",
        instance_id=qhana_instance.id,
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
def check_task_status(self, url: str, instance_id: str, camunda_external_task: str):
    external_task: ExternalTask = ExternalTask.from_dict(camunda_external_task)

    # TODO: Timeout if no result after a long time
    try:
        contents = request_json(url)

        # TODO: handle empty content better
        if contents is None:
            raise ValueError

        qhana_instance_status = contents["status"]
    except ConnectionError or ValueError:
        TASK_LOGGER.warning(
            f"QHAna plugin result under '{url}' could not be reached. Throwing BPMN exception...",
            exc_info=True,
        )

        camunda_client = get_camunda_client()

        # Throw bpmn error
        camunda_client.external_task_bpmn_error(
            task=external_task,
            error_code="qhana-plugin-unreachable",
            error_message="QHAna plugin could not be invoked.",
        )
        return

    # Check if qhana task completed successfully
    if qhana_instance_status == "SUCCESS":
        TASK_LOGGER.info(f"QHAna instance with id: {instance_id} completed successfully")

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
            f"QHAna instance with id: {instance_id} failed. Throwing BPMN exception..."
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
