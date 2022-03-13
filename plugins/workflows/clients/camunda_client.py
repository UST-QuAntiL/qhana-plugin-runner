import logging
import uuid
from typing import Optional
import requests
from ..util.bpmn_handler import BpmnHandler
from ..clients.qhana_task_client import QhanaTaskClient
from ..datatypes.camunda_datatypes import Deployment, ProcessInstance, ExternalTask
from ..listeners.camunda_process_end_listener import CamundaProcessEndListener
from ..listeners.external_task_listener import ExternalTaskListener
from ..listeners.qhana_plugin_end_listener import QhanaPluginEndListener
from ..util.helper import endpoint_found

logger = logging.getLogger(__name__)


class CamundaClient:
    """
    Handles setup for deployments, process instances and listeners.
    Removes deployment after instance has finished
    """

    def __init__(
            self,
            bpmn_handler: BpmnHandler = None,
            qhana_task_client: QhanaTaskClient = None,
            base_url: str = None,
            plugin_prefix: str = "plugin",
            poll_interval=5000
    ):
        self.bpmn_handler = bpmn_handler
        self.qhana_task_client = qhana_task_client
        self.m_base_url = base_url
        self.plugin_prefix = plugin_prefix
        self.m_poll_interval = poll_interval
        self.workerId = str(uuid.uuid4())
        self.deployment: Optional[Deployment] = None
        self.process_instance: Optional[ProcessInstance] = None
        self.qhana_listener: Optional[QhanaPluginEndListener] = None
        self.external_task_listener: Optional[ExternalTaskListener] = None
        self.process_end_listener: Optional[CamundaProcessEndListener] = None
        self.process_end = False

        logger.info(f"New camunda client created with workerId: {self.workerId}")

    def base_url(self, url):
        """
        :param url: The camunda endpoint used for all worker related requests
        :return:
        """
        self.m_base_url = url
        return self

    def poll_interval(self, timeout):
        """
        Sets the poll interval for the Camunda REST API
        :param timeout: The interval in milliseconds
        :return:
        """
        self.m_poll_interval = timeout
        return self

    def resource(self, bpmn: BpmnHandler):
        """
        :param bpmn: BPMN model for deployment
        :return:
        """
        self.bpmn_handler = bpmn
        return self

    def build(self):
        """
        Deploys the model, creates a model instance and polls for external tasks in the specified interval.
        After the instance terminates the model is undeployed
        :return:
        """
        self.deploy()
        self.create_instance()
        self.external_task_listener = ExternalTaskListener(self, callback=self.qhana_task_client.create_qhana_plugin_instances).start()
        self.qhana_listener = QhanaPluginEndListener(self, callback=self.qhana_task_client.complete_qhana_task).start()
        self.process_end_listener = CamundaProcessEndListener(self, callback=self.stop).start()
        return self

    def lock(self, task: ExternalTask):
        """
        Locks an external task so that the worker can use it
        :param task: The task to be locked
        :return:
        """
        # TODO: Instead of just setting a high lock duration, automatically re-lock when expired
        response = requests.post(f"{self.m_base_url}/external-task/{task.id}/lock",
                                 json={"workerId": self.workerId, "lockDuration": "999999999"})
        endpoint_found(response)

    def external_task_bpmn_error(self, task: ExternalTask, error_code: str, error_message: str):
        """
        Throws a bpmn error
        :param task: The failed task
        :param error_code: Code for exception
        :param error_message: Description of the exception
        :return:
        """

        output_variables = {
            "output": {
                "value": "Exception thrown.",
                "type": "String"
            }
        }
        response = requests.post(f"{self.m_base_url}/external-task/{task.id}/bpmnError",
                                 json={"workerId": self.workerId, "errorCode": error_code,
                                       "errorMessage": error_message, "variables": output_variables})
        endpoint_found(response)

    def complete_task(self, task: ExternalTask, result: dict):
        """
        External task is removed from the external tasks list from the workflow instance
        :param result: Return values for the external task
        :param task: The completed task
        :return:
        """
        response = requests.post(f"{self.m_base_url}/external-task/{task.id}/complete",
                                 json={"workerId": self.workerId, "variables": result})
        endpoint_found(response)

    def get_task_local_variables(self, task: ExternalTask):
        """
        Gets all local variables of an external task
        :param task: The external task
        :return:
        """
        response = requests.get(f"{self.m_base_url}/execution/{task.execution_id}/localVariables")
        if endpoint_found(response):
            return response.json()

    def get_global_variable(self, name: str):
        """
        Retrieves the global variable for a given name
        :param name: The global variable name
        :return:
        """
        response = requests.get(f"{self.m_base_url}/process-instance/{self.process_instance.id}/variables/{name}")
        if endpoint_found(response):
            return response.json()["value"]

    def get_task_execution_id(self, task_id: str):
        """
        Gets the execution id of an external task
        :param task_id: The task to get the execution id for
        :return:
        """
        response = requests.get(f"{self.m_base_url}/external-task/{task_id}")
        if endpoint_found(response):
            return response.json()["executionId"]

    def deploy(self):
        """
        Deploy a BPMN model to the Camunda Engine
        :return:
        """
        response = requests.post(f"{self.m_base_url}/deployment/create",
                                 files={self.bpmn_handler.filename: self.bpmn_handler.bpmn})
        if endpoint_found(response):
            self.deployment = Deployment.deserialize(response.json())

    def create_instance(self):
        """
        Create a workflow instance from the deployed BPMN model
        :return:
        """
        response = requests.post(f"{self.m_base_url}/process-definition/{self.deployment.process_definition_id}/start",
                                 json={"variables": {}})
        if endpoint_found(response):
            self.process_instance = ProcessInstance.deserialize(response.json())

    def is_process_active(self):
        """
        Checks if the process is still in the process list and thus active
        :return:
        """
        response = requests.get(f"{self.m_base_url}/process-instance")
        if endpoint_found(response):
            for process in response.json():
                if process["id"] == self.process_instance.id:
                    self.process_instance = ProcessInstance.deserialize(process)
                    return True
            return False

    def wait_for_process_end(self):
        """
        Waits for process end (blocking)
        :return:
        """
        while not self.process_end:
            pass

        return

    def undeploy(self, cascade: bool):
        """
        Deletes the active deployment
        :param cascade: Deletes all potentially active workflow instances
        Applying cascade to json doesn't work for some reason, multiple bug reports on the camunda forum open..
        :return:
        """
        response = requests.delete(f"{self.m_base_url}/deployment/{self.deployment.id}?cascade={str(cascade).lower()}")
        try:
            endpoint_found(response)
        except:
            return

    def stop(self):
        """
        Stops the camunda client and signals end
        :return:
        """
        self.undeploy(cascade=True)
        self.process_end = True
