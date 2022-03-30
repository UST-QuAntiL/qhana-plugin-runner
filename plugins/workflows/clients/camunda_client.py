import datetime
import logging
import requests
from celery.utils.log import get_task_logger

from .. import conf as config
from typing import Optional
from ..util.bpmn_handler import BpmnHandler
from ..datatypes.camunda_datatypes import Deployment, ProcessInstance, ExternalTask, CamundaConfig
from ..util.helper import endpoint_found, endpoint_found_simple, request_json
from dateutil import parser

logger = logging.getLogger(__name__)


class CamundaClient:
    """
    Handles setup for deployments, process instances and listeners.
    Removes deployment after instance has finished
    """

    def __init__(
        self,
        camunda_config: CamundaConfig,
        bpmn_handler: Optional[BpmnHandler] = None,
    ):
        self.camunda_config = camunda_config
        self.bpmn_handler = bpmn_handler

    def base_url(self, url):
        """
        :param url: The camunda endpoint used for all worker related requests
        :return:
        """
        self.camunda_config.base_url = url
        return self

    def poll_interval(self, timeout):
        """
        Sets the poll interval for the Camunda REST API
        :param timeout: The interval in milliseconds
        :return:
        """
        self.camunda_config.poll_interval = timeout
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
        return self

    def lock(self, task: ExternalTask):
        """
        Locks an external task so that the worker can use it
        :param task: The task to be locked
        :return:
        """
        # TODO: Instead of just setting a high lock duration, automatically re-lock when expired
        response = requests.post(f"{self.camunda_config.base_url}/external-task/{task.id}/lock",
                                 json={"workerId": self.camunda_config.worker_id, "lockDuration": "999999999"})
        endpoint_found(response)

    def is_locked(self, task: ExternalTask):
        response = requests.get(f"{self.camunda_config.base_url}/external-task/{task.id}")
        if endpoint_found(response):
            task_data = response.json()

            if task_data["lockExpirationTime"] is None:
                return False

            lock_expiration_time = datetime.datetime.strptime(task_data["lockExpirationTime"], "%Y-%m-%dT%H:%M:%S.%f%z")
            current_time = parser.parse(datetime.datetime.now().astimezone().replace(microsecond=0).isoformat())

            if lock_expiration_time > current_time:
                return True

            return False

    def external_task_bpmn_error(self, task: ExternalTask, error_code: str, error_message: str):
        """
        Throws a bpmn error
        :param task: The failed task
        :param error_code: Code for exception
        :param error_message: Description of the exception
        :return:
        """

        # Completing an external task always requires specifying output variables. This output variable should not be
        # used, instead refer to error_code and error_message for further exception handling.
        output_variables = {
            "output": {
                "value": "Exception thrown.",
                "type": "String"
            }
        }
        response = requests.post(f"{self.camunda_config.base_url}/external-task/{task.id}/bpmnError",
                                 json={"workerId": self.camunda_config.worker_id, "errorCode": error_code,
                                       "errorMessage": error_message, "variables": output_variables})
        endpoint_found(response)

    def complete_task(self, task: ExternalTask, result: dict):
        """
        External task is removed from the external tasks list from the workflow instance
        :param result: Return values for the external task
        :param task: The completed task
        :return:
        """
        response = requests.post(f"{self.camunda_config.base_url}/external-task/{task.id}/complete",
                                 json={"workerId": self.camunda_config.worker_id, "variables": result})
        endpoint_found(response)

    def complete_human_task(self, human_task_id: str, result: dict):
        response = requests.post(f"{self.camunda_config.base_url}/task/{human_task_id}/complete",
                                 json={"workerId": self.camunda_config.worker_id, "variables": result})
        endpoint_found(response)

    def get_task_local_variables(self, task: ExternalTask):
        """
        Gets all local variables of an external task
        :param task: The external task
        :return:
        """
        response = requests.get(f"{self.camunda_config.base_url}/execution/{task.execution_id}/localVariables")
        if endpoint_found(response):
            return response.json()

    def get_global_variable(self, name: str, external_task: ExternalTask):
        """
        Retrieves the global variable for a given name
        :param name: The global variable name
        :param external_task
        :return:
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/process-instance/{external_task.process_instance_id}/variables/{name}")
        if endpoint_found(response):
            return response.json()["value"]

    def get_instance_return_variables(self):
        variables = request_json(
            f"{self.camunda_config.base_url}/process-instance/{self.camunda_config.process_instance.id}/variables")
        return_variables = []

        for variable_key in variables.keys():
            if variable_key.startswith(config['workflow_out']['prefix']):
                return_variables.append(
                    {
                        variable_key: variables[variable_key]
                    }
                )

        return return_variables

    def get_task_execution_id(self, task_id: str):
        """
        Gets the execution id of an external task
        :param task_id: The task to get the execution id for
        :return:
        """
        response = requests.get(f"{self.camunda_config.base_url}/external-task/{task_id}")
        if endpoint_found(response):
            return response.json()["executionId"]

    def get_human_task_rendered_form(self, task_id: str):
        """
        TODO: Docs
        """
        response = requests.get(f"{self.camunda_config.base_url}/task/{task_id}/rendered-form")
        if endpoint_found_simple(response):
            return response.text

    def deploy(self):
        """
        Deploy a BPMN model to the Camunda Engine
        :return:
        """
        response = requests.post(f"{self.camunda_config.base_url}/deployment/create",
                                 files={self.bpmn_handler.filename: self.bpmn_handler.bpmn})
        if endpoint_found(response):
            self.camunda_config.deployment = Deployment.deserialize(response.json())

    def create_instance(self):
        """
        Create a workflow instance from the deployed BPMN model
        :return:
        """
        response = requests.post(
            f"{self.camunda_config.base_url}/process-definition/{self.camunda_config.deployment.process_definition_id}/start",
            json={"variables": {}})
        if endpoint_found(response):
            self.camunda_config.process_instance = ProcessInstance.deserialize(response.json())

    def is_process_active(self):
        """
        Checks if the process is still in the process list and thus active
        :return:
        """
        response = requests.get(f"{self.camunda_config.base_url}/process-instance")
        if endpoint_found(response):
            for process in response.json():
                if process["id"] == self.camunda_config.process_instance.id:
                    self.camunda_config.process_instance = ProcessInstance.deserialize(process)
                    return True
            return False

    def undeploy(self, cascade: bool):
        """
        Deletes the active deployment
        :param cascade: Deletes all potentially active workflow instances
        Applying cascade to json doesn't work for some reason, multiple bug reports on the camunda forum open..
        :return:
        """
        response = requests.delete(
            f"{self.camunda_config.base_url}/deployment/{self.camunda_config.deployment.id}?cascade={str(cascade).lower()}")
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
