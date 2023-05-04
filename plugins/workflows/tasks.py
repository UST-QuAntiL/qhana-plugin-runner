import json
from os import PathLike
from pathlib import Path
from typing import Mapping, Optional

from celery.utils.log import get_task_logger
from requests.exceptions import ConnectionError, HTTPError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from . import Workflows
from .clients.camunda_client import CamundaClient, CamundaManagementClient
from .exceptions import CamundaClientError, CamundaServerError, WorkflowNotFoundError
from .schemas import InputParameters, WorkflowsParametersSchema

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{Workflows.instance.identifier}.start_workflow_with_arguments",
    bind=True,
    autoretry_for=(ConnectionError, CamundaServerError),
    retry_backoff=True,
    max_retries=10,
)
def start_workflow_with_arguments(self, db_id: int, workflow_id: str) -> None:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(f"Could not load task data for task id '{db_id}'!")
        raise ValueError(f"Task {db_id} was not found!")

    workflow_inputs: Mapping = json.loads(
        task_data.parameters if task_data.parameters else "{}"
    )

    camunda = CamundaManagementClient(config)

    try:
        process_instance = camunda.start_workflow(
            definition_id=workflow_id, form_inputs=workflow_inputs
        )
    except HTTPError as err:
        print(err.request.headers)
        if err.response is not None and err.response.status_code:
            response_status: int = err.response.status_code
            if response_status == 404:
                raise WorkflowNotFoundError(
                    message=f"Workflow {workflow_id} could not be found!"
                ) from err
            if 400 <= response_status < 500:
                TASK_LOGGER.warning(
                    "Starting the workflow failed because of a client error. (Message body: {{}})",
                    err.response.text,
                    exc_info=False,
                )
                raise CamundaClientError(
                    message="Workflow start received unprocessable entities and could not proceed."
                ) from err
            if 500 <= response_status < 600:
                raise CamundaServerError(
                    message="Starting the workflow failed because of a server error."
                ) from err
        TASK_LOGGER.warning(
            f"Starting the workflow failed because of an unknown error. (Message body: {err.response.text})",
            exc_info=False,
        )
        raise
    TASK_LOGGER.info(f"Started process instance with id '{process_instance['id']}'!")

    assert isinstance(task_data.data, dict)

    # Set the process instance id used to create Camunda configs in sub-steps
    task_data.data["camunda_process_instance_id"] = process_instance["id"]
    task_data.data["camunda_process_definition_id"] = workflow_id
    task_data.add_task_log_entry("Started workflow.")
    task_data.save(commit=True)


# TODO remove old start workflow task
@CELERY.task(name=f"{Workflows.instance.identifier}.start_workflow", bind=True)
def start_workflow(self, db_id: int) -> None:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(f"Could not load task data for task id '{db_id}'!")
        raise ValueError(f"Task {db_id} was not found!")

    param_schema = WorkflowsParametersSchema()

    input_params: InputParameters = param_schema.loads(task_data.parameters)

    # BPMN file
    bpmn_folder: PathLike | None = config.get("workflow_folder")

    if bpmn_folder:
        bpmn_folder = Path(bpmn_folder)
        bpmn_path = bpmn_folder / (Path("test.bpmn").with_name(input_params.input_bpmn))
        bpmn_path = bpmn_path.with_suffix(".bpmn").resolve()

        if not bpmn_path.exists():
            builtin_path = Path(__file__).parent / Path("bpmn")
            bpmn_path = builtin_path / (
                Path("test.bpmn").with_name(input_params.input_bpmn)
            )
            bpmn_path = bpmn_path.with_suffix(".bpmn").resolve()

        if not bpmn_path.exists() or not bpmn_path.is_file():
            TASK_LOGGER.error(
                f"Requested BPMN file '{input_params.input_bpmn}' does not exist."
            )
            raise ValueError("BPMN file does not exist!")
    else:
        TASK_LOGGER.error(f"No folder configured to search for BPMN files.")
        raise ValueError("BPMN file folder is not configured!")

    # Client
    camunda_client = CamundaClient(config)

    # Deploy BPMN file and create workflow instance
    process_definition_id = camunda_client.deploy(bpmn_path)
    process_instance_id = camunda_client.create_instance(
        process_definition_id=process_definition_id
    )

    assert isinstance(task_data.data, dict)
    # Set the process instance id used to create Camunda configs in sub-steps
    task_data.data["camunda_process_instance_id"] = process_instance_id
    task_data.data["camunda_process_definition_id"] = process_definition_id
    task_data.save(commit=True)

    TASK_LOGGER.info(f"Started new workflow task with db_id: '{db_id}'")


@CELERY.task(name=f"{Workflows.instance.identifier}.process_input", bind=True)
def process_input(self, db_id: int) -> None:
    TASK_LOGGER.info(f"Started input processing with db_id: '{db_id}'")

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(f"Could not load task data for task id '{db_id}'!")
        raise ValueError(f"Task {db_id} was not found!")

    assert isinstance(task_data.data, dict)

    input_params: dict = json.loads(task_data.parameters)
    human_task_id = task_data.data["human_task_id"]

    # Client
    camunda_client = CamundaClient(config)

    variables = {
        key: {
            "value": val,
        }
        for key, val in input_params.items()
    }

    camunda_client.complete_human_task(human_task_id, variables)

    task_data.clear_previous_step()
    task_data.save(commit=True)
