from celery.utils.log import get_task_logger
from plugins.workflows import Workflows
from plugins.workflows.clients.camunda_client import CamundaClient
from plugins.workflows.config import CAMUNDA_BASE_URL, CAMUNDA_GENERAL_POLL_TIMEOUT
from plugins.workflows.datatypes.camunda_datatypes import CamundaConfig, ExternalTask
from plugins.workflows.util.helper import request_json
from plugins.workflows.watchers.external.qhana_instance_watcher import qhana_instance_watcher
from qhana_plugin_runner.celery import CELERY

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Workflows.instance.identifier}.external.camunda_task_watcher", ignore_result=True)
def camunda_task_watcher():
    """
    Watches for new Camunda external task. For each new task found a qhana_watcher celery task is spawned.
    """

    # Client
    camunda_client = CamundaClient(CamundaConfig(base_url=CAMUNDA_BASE_URL, poll_interval=CAMUNDA_GENERAL_POLL_TIMEOUT))

    external_tasks = request_json(f"{camunda_client.camunda_config.base_url}/external-task")
    external_tasks = [ExternalTask.deserialize(external_task) for external_task in external_tasks]

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
            # Spawn new watcher for the external task
            qhana_instance_watcher.s(external_task).delay()
