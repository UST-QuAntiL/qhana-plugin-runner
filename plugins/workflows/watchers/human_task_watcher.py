import random
import requests
from typing import Optional
from celery.utils.log import get_task_logger
from plugins.workflows import Workflows
from plugins.workflows.clients.camunda_client import CamundaClient
from plugins.workflows.datatypes.camunda_datatypes import CamundaConfig, HumanTask
from plugins.workflows.util.helper import request_json
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Workflows.instance.identifier}.human_task_watcher", bind=True)
def human_task_watcher(self, db_id: int, camunda_config: dict) -> None:
    # Client
    camunda_client = CamundaClient(CamundaConfig.from_dict(camunda_config))

    # Get all Camunda human tasks
    human_tasks = request_json(f"{camunda_client.camunda_config.base_url}/task")

    for human_task in human_tasks:
        human_task = HumanTask.deserialize(human_task)

        # Check if Human Task is open and part of the current workflow instance
        if (human_task.delegation_state == "PENDING" or human_task.delegation_state is None) and \
            human_task.process_instance_id == camunda_client.camunda_config.process_instance.id:
            form_variables = requests.get(
                f"{camunda_client.camunda_config.base_url}/task/{human_task.id}/form-variables")

            task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

            task_data.data["form_params"] = str(form_variables.json())
            task_data.data["human_task_id"] = human_task.id
            task_data.save(commit=True)

            # TODO: Remove
            href = f"http://localhost:5005/plugins/workflows@v0-1-1/{db_id}/demo-step-process/"
            ui_href = f"http://localhost:5005/plugins/workflows@v0-1-1/{db_id}/demo-step-ui/"

            # For testing purposes
            sid = f"{human_task.id}.{random.randrange(1, 100_000_000)}"

            # Add new sub-step task for input gathering
            new_sub_step_task = add_step.s(
                db_id=db_id, step_id=sid, href=href, ui_href=ui_href, prog_value=42,
                task_log="",
            )

            new_sub_step_task.link_error(save_task_error.s(db_id=db_id))
            new_sub_step_task.apply_async()

            TASK_LOGGER.info(f"Started step..  {sid}")

            return
