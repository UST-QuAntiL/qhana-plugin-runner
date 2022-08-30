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
from ..util.helper import get_form_variables, request_json

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Workflows.instance.identifier}.run_human_task_watcher", bind=True)
def run_human_task_watcher(self, db_id: int, camunda_config: dict) -> None:
    # Config
    camunda_config = CamundaConfig.from_dict(camunda_config)
    # Client
    camunda_client = CamundaClient(camunda_config)

    # Spawn new human task watcher if workflow instance is still active
    if camunda_client.is_process_active():
        human_task_watcher.s(db_id, camunda_config.to_dict()).apply_async(
            countdown=config["polling_rates"]["external_watcher"]
        )

        TASK_LOGGER.info(f"Started human task watcher with db_id: '{db_id}'")


@CELERY.task(name=f"{Workflows.instance.identifier}.human_task_watcher", bind=True)
def human_task_watcher(self, db_id: int, camunda_config: dict) -> None:
    # Client
    camunda_client = CamundaClient(CamundaConfig.from_dict(camunda_config))

    # Get all Camunda human tasks
    human_tasks = request_json(f"{camunda_client.camunda_config.base_url}/task")

    for human_task in human_tasks:
        human_task = HumanTask.deserialize(human_task)

        # Check if Human Task is open and part of the current workflow instance
        if (
            (
                human_task.delegation_state == "PENDING"
                or human_task.delegation_state is None
            )
            and human_task.process_instance_id
            == camunda_client.camunda_config.process_instance.id
        ):

            # Save result file with workflow instance variables marked as output
            if human_task.name == config["workflow_out"]["camunda_user_task_name"]:
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
                    task_log="",
                    db_id=db_id,
                ).delay()

                return

            rendered_form = camunda_client.get_human_task_rendered_form(human_task.id)
            instance_variables = requests.get(
                f"{camunda_client.camunda_config.base_url}/task/{human_task.id}/form-variables"
            ).json()

            # Extract form variables from the rendered form. Cannot use camunda endpoint for form variables (broken)
            form_variables = get_form_variables(rendered_form, instance_variables)

            process_definition_id = (
                camunda_client.camunda_config.process_instance.definition_id
            )

            bpmn_properties = {
                "bpmn_xml_url": f"{camunda_client.camunda_config.base_url}/process-definition/{process_definition_id}/xml",
                "human_task_definition_key": human_task.task_definition_key,
            }

            task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
            task_data.data["form_params"] = str(form_variables)
            task_data.data["bpmn"] = str(bpmn_properties)
            task_data.data["human_task_id"] = human_task.id
            task_data.save(commit=True)

            # TODO: Remove
            href = f"http://localhost:5005/plugins/workflows@v0-5-1/{db_id}/human-task-process/"
            ui_href = (
                f"http://localhost:5005/plugins/workflows@v0-5-1/{db_id}/human-task-ui/"
            )

            # Add new sub-step task for input gathering
            new_sub_step_task = add_step.s(
                db_id=db_id,
                step_id=human_task.id,
                href=href,
                ui_href=ui_href,
                prog_value=42,
                task_log="",
            )

            new_sub_step_task.link_error(save_task_error.s(db_id=db_id))
            new_sub_step_task.apply_async()

            TASK_LOGGER.info(f"Started step...")

            return

    run_human_task_watcher.apply_async(
        (db_id, camunda_config),
        countdown=config["polling_rates"]["external_watcher"],
    )
