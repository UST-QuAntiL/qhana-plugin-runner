from celery.utils.log import get_task_logger
from plugins.workflows import Workflows
from plugins.workflows.clients.camunda_client import CamundaClient
from plugins.workflows.clients.qhana_task_client import QhanaTaskClient
from plugins.workflows.config import CAMUNDA_BASE_URL, CAMUNDA_GENERAL_POLL_TIMEOUT, QHANA_PLUGIN_ENDPOINTS
from plugins.workflows.datatypes.camunda_datatypes import ExternalTask, CamundaConfig
from plugins.workflows.datatypes.qhana_datatypes import QhanaResult, QhanaOutput
from plugins.workflows.util.helper import request_json
from qhana_plugin_runner.celery import CELERY

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Workflows.instance.identifier}.external.qhana_instance_watcher", ignore_result=True)
def qhana_instance_watcher(camunda_external_task: str):
    """
    Creates new qhana plugin instances and watches for results.
    """

    # Clients
    camunda_client = CamundaClient(CamundaConfig(base_url=CAMUNDA_BASE_URL, poll_interval=CAMUNDA_GENERAL_POLL_TIMEOUT))
    qhana_client = QhanaTaskClient(QHANA_PLUGIN_ENDPOINTS)

    # Deserialize
    external_task: ExternalTask = ExternalTask.from_dict(camunda_external_task)

    # Create new qhana plugin instance
    qhana_instance = qhana_client.create_qhana_plugin_instance(camunda_client, external_task)

    TASK_LOGGER.info(f"Created QHAna plugin instance {qhana_instance.plugin.name} with id: {qhana_instance.id}")

    # TODO: Timeout if no result after a long time
    while True:
        contents = request_json(f"{qhana_instance.plugin.api_endpoint}/tasks/{qhana_instance.id}")

        qhana_instance_status = contents["status"]

        # Check if qhana task completed successfully
        if qhana_instance_status == "SUCCESS":
            TASK_LOGGER.info(f"QHAna instance with id: {qhana_instance.id} completed successfully")

            outputs = [QhanaOutput.deserialize(output) for output in contents["outputs"]]
            qhana_result = QhanaResult(qhana_instance, outputs)

            # Complete external task with qhana task result
            qhana_client.complete_qhana_task(camunda_client, qhana_result)

            return
        elif qhana_instance_status == "FAILURE":
            TASK_LOGGER.info(f"QHAna instance with id: {qhana_instance.id} failed. Throwing BPMN exception...")

            # Throw bpmn error
            camunda_client.external_task_bpmn_error(
                task=qhana_instance.external_task,
                error_code="qhana-plugin-failure",
                error_message="QHAna plugin failed execution."
            )

            return
