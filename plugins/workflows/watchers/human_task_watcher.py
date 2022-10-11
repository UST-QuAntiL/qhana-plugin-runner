import json
from tempfile import SpooledTemporaryFile
from typing import Optional

from celery.utils.log import get_task_logger
from requests.exceptions import RequestException

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result

from .. import Workflows
from ..exceptions import WorkflowStoppedError
from ..clients.camunda_client import CamundaClient
from ..datatypes.camunda_datatypes import CamundaConfig

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


class NoHumanTaskError(Exception):
    pass


def persist_workflow_output(
    db_task: ProcessingTask, camunda_client: CamundaClient, process_instance_id: str
):
    db_task.add_task_log_entry("Workflow process instance finished successfully.")
    return_variables = camunda_client.get_historic_process_instance_variables(
        process_instance_id
    )

    data_output_keys = {"name", "contentType", "dataType", "href"}
    data_outputs = []

    returned_entities = []
    for var in return_variables:
        value = var.get("value")
        name: str = var["name"]
        while (
            name.startswith((f'{config["workflow_out"]["prefix"]}.', "qoutput."))
            and "." in name  # this prevents infinite loops!
        ):
            name = name.split(".", maxsplit=1)[-1]
        returned_entities.append(
            {
                "ID": var["id"],
                "href": f'{camunda_client.camunda_config.base_url}/history/variable-instance/{var["id"]}',
                "name": name,
                "value": value,
                "processDefinitionId": var["processDefinitionId"],
                "processInstanceId": var["processInstanceId"],
                "executionId": var["executionId"],
                "activityInstanceId": var["activityInstanceId"],
                "rootProcessInstanceId": var["rootProcessInstanceId"],
            }
        )
        if isinstance(value, dict):
            if value.keys() >= data_output_keys:
                data_outputs.append(value)
        if isinstance(value, (list, tuple)):
            for val in value:
                if val.keys() >= data_output_keys:
                    data_outputs.append(val)

    for output in data_outputs:
        STORE.persist_task_result(
            db_task.id,
            file_=output["href"],
            file_type=output["dataType"],
            mimetype=output["contentType"],
            file_name=output["name"],
            storage_provider="url_file_store",
            commit=False,
        )

    with SpooledTemporaryFile(mode="wt") as result:
        save_entities(returned_entities, result, "application/X-lines+json")
        STORE.persist_task_result(
            db_task.id,
            file_=result,
            file_name="workflow-output.json",
            file_type="entity/stream",
            mimetype="application/X-lines+json",
            commit=True,
        )


@CELERY.task(
    name=f"{Workflows.instance.identifier}.human_task_watcher",
    bind=True,
    ignore_result=True,
    autoretry_for=(NoHumanTaskError,),
    retry_backoff=True,
    max_retries=None,
)
def human_task_watcher(self, db_id: int) -> None:
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(db_task.data, dict)

    process_instance_id: str = db_task.data["camunda_process_instance_id"]

    # Client
    camunda_client = CamundaClient(CamundaConfig.from_config(config))

    # Spawn new human task watcher if workflow instance is still active
    if not camunda_client.is_process_active(process_instance_id):
        if not camunda_client.has_process_completed(process_instance_id):
            db_task.add_task_log_entry(
                "Workflow process instance was stopped unexpectedly."
            )
            db_task.save(commit=True)
            raise WorkflowStoppedError(
                "Workflow process instance was stopped unexpectedly."
            )
        persist_workflow_output(db_task, camunda_client, process_instance_id)

        save_task_result.s(
            task_log="Stored workflow output.",
            db_id=db_id,
        ).delay()
        return

    # Get all Camunda human tasks
    try:
        human_tasks = camunda_client.get_human_tasks(
            process_instance_id=process_instance_id
        )
    except RequestException as err:
        TASK_LOGGER.info(f"Exception while retrieving human tasks: {err}")
        # camunda could not be reached/produced an error => retry later
        raise NoHumanTaskError  # retry

    TASK_LOGGER.debug(f"Human Tasks: {human_tasks}")

    for human_task in human_tasks:
        assert (
            human_task.process_instance_id == process_instance_id
        ), "camunda client returned a human task for a different process instance!"
        if human_task.delegation_state not in ("PENDING", None):
            continue  # task is already taken

        # Extract form variables from the rendered form. Cannot use camunda endpoint for form variables (broken)
        form_variables = camunda_client.get_human_task_form_variables(
            human_task_id=human_task.id
        )

        process_definition_id = db_task.data["camunda_process_definition_id"]

        bpmn_properties = {
            "bpmn_xml_url": f"{camunda_client.camunda_config.base_url}/process-definition/{process_definition_id}/xml",
            "human_task_definition_key": human_task.task_definition_key,
        }

        db_task.add_task_log_entry(f"Found new human task '{human_task.id}'.")
        db_task.data["form_params"] = json.dumps(form_variables)
        db_task.data["bpmn"] = json.dumps(bpmn_properties)
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
