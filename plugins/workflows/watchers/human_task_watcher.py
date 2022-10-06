import json
from tempfile import SpooledTemporaryFile
from typing import Optional

import requests
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result

from .. import Workflows
from ..clients.camunda_client import CamundaClient
from ..datatypes.camunda_datatypes import CamundaConfig, HumanTask
from ..util.helper import get_form_variables

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


class NoHumanTaskError(Exception):
    pass


@CELERY.task(
    name=f"{Workflows.instance.identifier}.human_task_watcher",
    bind=True,
    ignore_result=True,
    autoretry_for=(NoHumanTaskError,),
    retry_backoff=True,
    max_retries=None,
)
def human_task_watcher(self, db_id: int, camunda_config: dict) -> None:
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # Client
    camunda_client = CamundaClient(CamundaConfig.from_dict(camunda_config))

    # Spawn new human task watcher if workflow instance is still active
    if not camunda_client.is_process_active():
        db_task.add_task_log_entry("Workflow process instance was stopped.")
        db_task.save(commit=True)  # FIXME store workflow output
        return

    # Get all Camunda human tasks
    response = requests.get(f"{camunda_client.camunda_config.base_url}/task")
    if response.status_code != 200:
        # camunda could not be reached/produced an error => retry later
        raise NoHumanTaskError  # retry
    human_tasks = response.json()

    for human_task in human_tasks:
        human_task = HumanTask.deserialize(human_task)
        if human_task.delegation_state not in ("PENDING", None):
            continue  # only consieder unfinished tasks

        # Check if Human Task is open and part of the current workflow instance
        if (
            human_task.process_instance_id
            != camunda_client.camunda_config.process_instance.id
        ):
            continue  # only consider human tasks that belong to the current workflow

        # Save result file with workflow instance variables marked as output
        if human_task.name == config["workflow_out"]["camunda_user_task_name"]:
            # FIXME reading workflow outputs should be done if workflow completes, not in a human task...
            return_variables = camunda_client.get_instance_return_variables()

            with SpooledTemporaryFile(mode="w") as result:
                result.write(json.dumps(return_variables, indent=4, sort_keys=True))
                STORE.persist_task_result(
                    db_id,
                    result,
                    "result.json",
                    "workflow-output",
                    "application/json",
                )

            save_task_result.s(
                task_log="Stored workflow output.",
                db_id=db_id,
            ).delay()

            return

        rendered_form = camunda_client.get_human_task_rendered_form(human_task.id)
        instance_variables = requests.get(
            f"{camunda_client.camunda_config.base_url}/task/{human_task.id}/form-variables"
        ).json()  # TODO add error handling

        # Extract form variables from the rendered form. Cannot use camunda endpoint for form variables (broken)
        form_variables = get_form_variables(rendered_form, instance_variables)

        process_definition_id = (
            camunda_client.camunda_config.process_instance.definition_id
        )

        bpmn_properties = {
            "bpmn_xml_url": f"{camunda_client.camunda_config.base_url}/process-definition/{process_definition_id}/xml",
            "human_task_definition_key": human_task.task_definition_key,
        }

        db_task.add_task_log_entry(f"Found new human task '{human_task.id}'.")
        db_task.data["form_params"] = str(form_variables)
        db_task.data["bpmn"] = str(bpmn_properties)
        db_task.data["human_task_id"] = human_task.id
        db_task.save(commit=True)

        # Add new sub-step task for input gathering
        new_sub_step_task = add_step.s(
            db_id=db_id,
            step_id=human_task.id,
            href=db_task.data["href"],
            ui_href=db_task.data["ui_href"],
            prog_value=42,
            task_log="",
        )

        new_sub_step_task.link_error(save_task_error.s(db_id=db_id))
        new_sub_step_task.apply_async()

        TASK_LOGGER.info(f"Started step...")
        return

    raise NoHumanTaskError  # retry
