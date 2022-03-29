import ast
import random
import requests
from typing import Optional
from celery.utils.log import get_task_logger
from plugins.workflows.clients.camunda_client import CamundaClient
from plugins.workflows.config import CAMUNDA_BASE_URL, CAMUNDA_GENERAL_POLL_TIMEOUT
from plugins.workflows.datatypes.camunda_datatypes import CamundaConfig, ProcessInstance
from plugins.workflows import Workflows
from plugins.workflows.schemas import WorkflowsParametersSchema, InputParameters
from plugins.workflows.util.bpmn_handler import BpmnHandler
from plugins.workflows.watchers.human_task_watcher import human_task_watcher
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{Workflows.instance.identifier}.start_workflow", bind=True)
def start_workflow(self, db_id: int) -> None:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    param_schema = WorkflowsParametersSchema()
    input_params: InputParameters = param_schema.loads(task_data.parameters)

    # BPMN file
    bpmn = BpmnHandler(f"{input_params.input_bpmn}.bpmn")

    # Config
    camunda_config = CamundaConfig(
        base_url=CAMUNDA_BASE_URL,
        poll_interval=CAMUNDA_GENERAL_POLL_TIMEOUT,
    )

    # Client
    camunda_client = CamundaClient(camunda_config, bpmn)
    # Deploy BPMN file and create workflow instance
    camunda_client.build()

    # Set the process instance id used to create Camunda configs in sub-steps
    task_data.data["camunda_process_instance_id"] = camunda_config.process_instance.id
    task_data.save(commit=True)

    run_human_task_watcher.s(db_id, camunda_config.to_dict()).delay()

    TASK_LOGGER.info(f"Started new workflow task with db_id: '{db_id}'")


@CELERY.task(name=f"{Workflows.instance.identifier}.run_human_task_watcher", bind=True)
def run_human_task_watcher(self, db_id: int, camunda_config: dict) -> None:
    # Config
    camunda_config = CamundaConfig.from_dict(camunda_config)
    # Client
    camunda_client = CamundaClient(camunda_config)

    # Spawn new human task watcher if workflow instance is still active
    if camunda_client.is_process_active():
        human_task_watcher.s(db_id, camunda_config.to_dict()).apply_async(countdown=CAMUNDA_GENERAL_POLL_TIMEOUT)

        TASK_LOGGER.info(f"Started human task watcher with db_id: '{db_id}'")


@CELERY.task(name=f"{Workflows.instance.identifier}.process_input", bind=True)
def process_input(self, db_id: int) -> None:
    TASK_LOGGER.info(f"Started input processing with db_id: '{db_id}'")

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    input_params: dict = ast.literal_eval(task_data.parameters)
    process_instance_id = task_data.data["camunda_process_instance_id"]
    human_task_id = task_data.data["human_task_id"]

    # Config
    camunda_config = CamundaConfig(
        base_url=CAMUNDA_BASE_URL,
        poll_interval=CAMUNDA_GENERAL_POLL_TIMEOUT,
        process_instance=ProcessInstance(process_instance_id),
    )

    # # Client
    # camunda_client = CamundaClient(CamundaConfig.from_dict(camunda_config))
    #
    # variables = {
    #     key: {
    #         "value": val,
    #     } for key, val in input_params.items()
    # }
    #
    # camunda_client.complete_human_task(human_task_id, variables)

    task_data.clear_previous_step()
    task_data.save(commit=True)

    run_human_task_watcher.s(
        db_id,
        camunda_config.to_dict()
    ).delay()
