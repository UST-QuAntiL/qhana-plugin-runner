import json
from tempfile import SpooledTemporaryFile
from typing import Optional

from celery.utils.log import get_task_logger
from requests.exceptions import RequestException

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_result

from .. import DeployWorkflow
from ..clients.camunda_client import CamundaClient
from ..exceptions import WorkflowStoppedError

config = DeployWorkflow.instance.config

TASK_LOGGER = get_task_logger(__name__)


class WorkflowNotFinished(Exception):
    pass


def persist_workflow_output(
    db_task: ProcessingTask, camunda_client: CamundaClient, process_instance_id: str
):
    db_task.add_task_log_entry("Workflow process instance finished successfully.")
    return_variables = camunda_client.get_historic_process_instance_variables(
        process_instance_id
    )

    has_return_variables = bool(return_variables)

    if not has_return_variables:
        # if no return variable is specified, then try to retrun all variables
        return_variables = camunda_client.get_all_historic_process_instance_variables(
            process_instance_id
        )

    data_output_keys = {"name", "contentType", "dataType", "href"}
    data_outputs = []

    returned_entities = []
    for name, var in return_variables.items():
        returned_entities.append(
            {
                "ID": var["id"],
                "href": f'{config["camunda_base_url"].rstrip("/")}/history/variable-instance/{var["id"]}',
                "name": name,
                "value": var["value"],
                "processDefinitionId": var["processDefinitionId"],
                "processInstanceId": var["processInstanceId"],
                "executionId": var["executionId"],
                "activityInstanceId": var["activityInstanceId"],
                "rootProcessInstanceId": var["rootProcessInstanceId"],
            }
        )
        value = var["value"]
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


def check_for_incidents(
    camunda_client: CamundaClient, process_instance_id: str, db_task: ProcessingTask
) -> bool:
    incidents = camunda_client.get_workflow_incidents(
        process_instance_id=process_instance_id
    )
    if not incidents:
        return False

    assert isinstance(db_task.data, dict)

    db_task.add_next_step(
        step_id="workflow-incident",
        href=db_task.data["href_incident"],
        ui_href=db_task.data["ui_href_incident"],
        commit=True,
    )

    return True


def check_for_human_tasks(
    camunda_client: CamundaClient, process_instance_id: str, db_task: ProcessingTask
) -> bool:
    try:
        human_tasks = camunda_client.get_human_tasks(
            process_instance_id=process_instance_id
        )
    except RequestException as err:
        TASK_LOGGER.info(f"Exception while retrieving human tasks: {err}")
        # camunda could not be reached/produced an error => retry later
        return False

    TASK_LOGGER.debug(f"Human Tasks: {human_tasks}")

    for human_task in human_tasks:
        assert (
            human_task.process_instance_id == process_instance_id
        ), "camunda client returned a human task for a different process instance!"
        if human_task.delegation_state not in ("PENDING", None):
            continue  # task is already taken

        # Extract form variables from the rendered form. Cannot use camunda endpoint for form variables (broken)
        form_variables = camunda_client.get_human_task_form_variables(
            human_task_id=human_task.id, form_key=human_task.form_key
        )

        assert isinstance(db_task.data, dict)

        db_task.add_task_log_entry(f"Found new human task '{human_task.id}'.")
        db_task.data["form_params"] = json.dumps(form_variables)
        db_task.data.pop("external_form_key", None)  # remove old form key
        if human_task.form_key and human_task.form_key.startswith("embedded:"):
            db_task.data["external_form_key"] = human_task.form_key
        db_task.data["human_task_id"] = human_task.id
        db_task.data["human_task_definition_key"] = human_task.task_definition_key

        db_task.add_next_step(
            step_id=human_task.id,
            href=db_task.data["href"],
            ui_href=db_task.data["ui_href"],
            commit=True,
        )

        return True

    return False


@CELERY.task(
    name=f"{DeployWorkflow.instance.identifier}.workflow_status_watcher",
    bind=True,
    ignore_result=True,
    autoretry_for=(WorkflowNotFinished,),
    retry_backoff=True,
    max_retries=None,
)
def workflow_status_watcher(self, db_id: int) -> None:
    db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if db_task is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    assert isinstance(db_task.data, dict)

    process_instance_id = db_task.data["camunda_process_instance_id"]
    assert isinstance(process_instance_id, str)

    # Client
    camunda_client = CamundaClient(config)

    # Check if the process is still active
    if not camunda_client.is_process_active(process_instance_id):
        if not camunda_client.has_process_completed(process_instance_id):
            db_task.add_task_log_entry(
                "Workflow process instance was stopped unexpectedly.",
                commit=True,
            )
            raise WorkflowStoppedError(
                "Workflow process instance was stopped unexpectedly."
            )
        TASK_LOGGER.info(
            f"Workflow finished, persisting task result. (Process instance id: {process_instance_id})"
        )
        persist_workflow_output(db_task, camunda_client, process_instance_id)

        self.replace(
            save_task_result.s(
                task_log="Stored workflow output.",
                db_id=db_id,
            )
        )
        return

    encountered_incident = check_for_incidents(
        camunda_client, process_instance_id, db_task
    )
    if encountered_incident:
        return  # do not check for further updates, wait for user response to sub-task first

    encountered_human_task = check_for_human_tasks(
        camunda_client, process_instance_id, db_task
    )
    if encountered_human_task:
        return  # do not check for further updates, wait for user response to sub-task first

    raise WorkflowNotFinished  # continue polling for updates
