from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY

from .qhana_instance_watcher import qhana_instance_watcher
from ... import Workflows
from ...clients.camunda_client import CamundaClient
from ...datatypes.camunda_datatypes import CamundaConfig, ExternalTask
from ...util.helper import request_json

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

    # Client
    camunda_client = CamundaClient(
        CamundaConfig(
            base_url=config["CAMUNDA_BASE_URL"],
            poll_interval=config["polling_rates"]["camunda_general"],
        )
    )

    external_tasks = request_json(
        f"{camunda_client.camunda_config.base_url}/external-task"
    )
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
            # unlock_task.s(camunda_external_task=external_task.to_dict()).apply_async()
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
            instance_task.link_error(unlock_task.si(camunda_external_task=external_task))
            instance_task.apply_async()


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

    camunda_client.unlock(external_task)
