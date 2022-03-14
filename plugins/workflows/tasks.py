from typing import Optional

from celery.utils.log import get_task_logger
from plugins.workflows import Workflows, WORKFLOWS_BLP
from plugins.workflows.clients.camunda_client import CamundaClient
from plugins.workflows.clients.qhana_task_client import QhanaTaskClient
from plugins.workflows.schemas import WorkflowsParametersSchema, InputParameters
from plugins.workflows.util.bpmn_handler import BpmnHandler
from plugins.workflows.util.result_store import ResultStore
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result

TASK_LOGGER = get_task_logger(__name__)
CAMUNDA_BASE_URL = "http://localhost:8080/engine-rest"
QHANA_PLUGIN_ENDPOINTS = ["http://localhost:5005/"]


@CELERY.task(name=f"{Workflows.instance.identifier}.entry", bind=True)
def run_model(self, db_id: int) -> str:

    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")

    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    param_schema = WorkflowsParametersSchema()
    input_params: InputParameters = param_schema.loads(task_data.parameters)

    model = BpmnHandler(f"{input_params.input_bpmn}.bpmn")

    result_store = ResultStore()
    qhana_task_client = QhanaTaskClient(QHANA_PLUGIN_ENDPOINTS, result_store)
    camunda_client = CamundaClient(db_id=db_id, task_data=task_data, qhana_task_client=qhana_task_client, poll_interval=5) \
        .base_url(CAMUNDA_BASE_URL) \
        .resource(model)
    camunda_client.build()
    camunda_client.wait_for_process_end()

    return "Workflow done."


@CELERY.task(name=f"{Workflows.instance.identifier}.step", bind=True)
def step(self, db_id: int) -> str:

    TASK_LOGGER.info(f"Starting new demo task with db id '{db_id}'")


    return "Step done."
