from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from requests.exceptions import ConnectionError

from ... import Workflows
from ...clients.camunda_client import CamundaClient
from ...clients.qhana_task_client import QhanaTaskClient
from ...datatypes.camunda_datatypes import CamundaConfig, ExternalTask
from ...datatypes.qhana_datatypes import QhanaOutput, QhanaResult
from ...util.helper import request_json

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{Workflows.instance.identifier}.external.qhana_instance_watcher",
    ignore_result=True,
)
def qhana_instance_watcher(camunda_external_task: str):
    """
    Creates new qhana plugin instances and watches for results.
    """
    TASK_LOGGER.info(f"Received task from camunda queue: {camunda_external_task}")

    # Clients
    camunda_client = CamundaClient(
        CamundaConfig(
            base_url=config["CAMUNDA_BASE_URL"],
            poll_interval=config["polling_rates"]["camunda_general"],
        )
    )
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

    # TODO: Timeout if no result after a long time
    while True:  # FIXME return control flow to worker periodically to free up resources
        try:
            contents = request_json(
                f"{qhana_instance.plugin.api_endpoint}/tasks/{qhana_instance.id}"
            )

            qhana_instance_status = contents["status"]
        except ConnectionError:
            TASK_LOGGER.warning(
                f"QHAna plugin result under '{qhana_instance.plugin.api_endpoint}/tasks/{qhana_instance.id}' could not be reached. Throwing BPMN exception...",
                exc_info=True,
            )

            # Throw bpmn error
            camunda_client.external_task_bpmn_error(
                task=external_task,
                error_code="qhana-plugin-unreachable",
                error_message="QHAna plugin could not be invoked.",
            )
            return

        # Check if qhana task completed successfully
        if qhana_instance_status == "SUCCESS":
            TASK_LOGGER.info(
                f"QHAna instance with id: {qhana_instance.id} completed successfully"
            )

            outputs = [QhanaOutput.deserialize(output) for output in contents["outputs"]]
            qhana_result = QhanaResult(qhana_instance, outputs)

            # Complete external task with qhana task result
            qhana_client.complete_qhana_task(camunda_client, qhana_result)

            return
        elif qhana_instance_status == "FAILURE":
            TASK_LOGGER.info(
                f"QHAna instance with id: {qhana_instance.id} failed. Throwing BPMN exception..."
            )

            # Throw bpmn error
            camunda_client.external_task_bpmn_error(
                task=qhana_instance.external_task,
                error_code="qhana-plugin-failure",
                error_message="QHAna plugin failed execution.",
            )

            return
