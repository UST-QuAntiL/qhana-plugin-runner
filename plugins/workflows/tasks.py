import json
from pathlib import Path
from typing import Optional

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from . import Workflows
from .clients.camunda_client import CamundaClient
from .datatypes.camunda_datatypes import CamundaConfig
from .schemas import InputParameters, WorkflowsParametersSchema
from .watchers.human_task_watcher import human_task_watcher

config = Workflows.instance.config

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Workflows.instance.identifier}.start_workflow", bind=True)
def start_workflow(self, db_id: int) -> None:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(f"Could not load task data for task id '{db_id}'!")
        raise ValueError(f"Task {db_id} was not found!")

    param_schema = WorkflowsParametersSchema()

    input_params: InputParameters = param_schema.loads(task_data.parameters)

    # BPMN file
    bpmn_folder: Path = config["WORKFLOW_FOLDER"]
    bpmn_path = bpmn_folder / (Path("test.bpmn").with_name(input_params.input_bpmn))
    bpmn_path = bpmn_path.with_suffix(".bpmn").resolve()

    if not bpmn_path.exists():
        builtin_path = Path(__file__).parent / Path("bpmn")
        bpmn_path = builtin_path / (Path("test.bpmn").with_name(input_params.input_bpmn))
        bpmn_path = bpmn_path.with_suffix(".bpmn").resolve()

    if not bpmn_path.exists() or not bpmn_path.is_file():
        TASK_LOGGER.error(
            f"Requested BPMN file '{input_params.input_bpmn}' does not exist."
        )
        raise ValueError("BPMN file does not exist!")

    # Client
    camunda_client = CamundaClient(CamundaConfig.from_config(config))

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
    camunda_client = CamundaClient(CamundaConfig.from_config(config))

    variables = {
        key: {
            "value": val,
        }
        for key, val in input_params.items()
    }

    camunda_client.complete_human_task(human_task_id, variables)

    task_data.clear_previous_step()
    task_data.save(commit=True)
