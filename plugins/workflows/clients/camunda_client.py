import datetime
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence

import requests
from requests.exceptions import HTTPError

from .. import Workflows
from ..exceptions import WorkflowDeploymentError
from ..datatypes.camunda_datatypes import CamundaConfig, ExternalTask, HumanTask

config = Workflows.instance.config

logger = logging.getLogger(__name__)


class CamundaClient:
    """
    Handles setup for deployments, process instances and listeners.
    Removes deployment after instance has finished
    """

    def __init__(
        self,
        camunda_config: CamundaConfig,
        timeout: int = config.get("request_timeout", 5 * 60),
    ):
        self.camunda_config = camunda_config
        self.timeout = timeout

    def lock(self, external_task_id: str):
        """
        Locks an external task so that the worker can use it
        :param task: The task to be locked
        """
        # TODO: Instead of just setting a high lock duration, automatically re-lock when expired
        response = requests.post(
            f"{self.camunda_config.base_url}/external-task/{external_task_id}/lock",
            json={"workerId": self.camunda_config.worker_id, "lockDuration": "999999999"},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def get_locked_external_tasks_count(self) -> int:
        response = requests.get(
            f"{self.camunda_config.base_url}/external-task/count",
            params={
                "workerId": self.camunda_config.worker_id,
                "locked": "true",
                "active": "true",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["count"]

    def get_external_tasks(self, limit: Optional[int] = None):
        """Get unlocked external tasks from the task queue.

        Args:
            limit (Optional[int], optional): limit the number of unlocked external tasks to fetch. If set at most limit external tasks will be returned. Defaults to None.

        Raises:
            ValueError: the limits have illegal values
        """
        params = {"active": "true", "notLocked": "true"}
        if limit is not None:
            if limit < 1:
                raise ValueError("The limit must not be smaller than 1!")
            params["maxResults"] = str(limit)
        response = requests.get(
            f"{self.camunda_config.base_url}/external-task",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        external_tasks = response.json()

        return [
            ExternalTask.deserialize(external_task) for external_task in external_tasks
        ]

    def unlock(self, external_task_id: str):
        """
        Unlocks an external task so that other workers can use it
        :param task: The task to be locked
        """
        response = requests.post(
            f"{self.camunda_config.base_url}/external-task/{external_task_id}/unlock",
            timeout=self.timeout,
        )
        response.raise_for_status()

    def check_lock_expired(self, lock_expiration_time: Optional[str]):
        if not lock_expiration_time:
            return True

        lock_time = datetime.datetime.strptime(
            lock_expiration_time, "%Y-%m-%dT%H:%M:%S.%f%z"
        )
        current_time = datetime.datetime.now(datetime.timezone.utc)

        return lock_time < current_time

    def is_locked(self, external_task_id: str):
        """
        Check if a task is locked
        :return: Locked status
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/external-task/{external_task_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()

        task_data = response.json()

        return not self.check_lock_expired(task_data["lockExpirationTime"])

    def external_task_bpmn_error(
        self, external_task_id: str, error_code: str, error_message: str
    ):
        """
        Throws a bpmn error
        :param external_task_id: The id of the failed task
        :param error_code: Code for exception
        :param error_message: Description of the exception
        """

        # Completing an external task always requires specifying output variables. This output variable should not be
        # used, instead refer to error_code and error_message for further exception handling.
        output_variables = {"output": {"value": "Exception thrown.", "type": "String"}}
        response = requests.post(
            url=f"{self.camunda_config.base_url}/external-task/{external_task_id}/bpmnError",
            json={
                "workerId": self.camunda_config.worker_id,
                "errorCode": error_code,
                "errorMessage": error_message,
                "variables": output_variables,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

    def complete_task(self, external_task_id: str, result: dict):
        """
        External task is removed from the external tasks list from the workflow instance
        :param external_task_id: The id of the completed task
        :param result: Return values for the external task
        """
        response = requests.post(
            f"{self.camunda_config.base_url}/external-task/{external_task_id}/complete",
            json={"workerId": self.camunda_config.worker_id, "variables": result},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def complete_human_task(self, human_task_id: str, result: dict):
        """
        Completes a human task with the user input as workflow variables
        :param human_task_id: Human task id
        :param result: User input result
        """
        response = requests.post(
            f"{self.camunda_config.base_url}/task/{human_task_id}/complete",
            json={"workerId": self.camunda_config.worker_id, "variables": result},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def get_human_tasks(self, process_instance_id: str) -> Sequence[HumanTask]:
        """Get all active human tasks of a given process instance."""
        response = requests.get(
            f"{self.camunda_config.base_url}/task",
            params={
                "processInstanceId": process_instance_id,
                "active": "true",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return [HumanTask.deserialize(t) for t in response.json()]

    def get_human_task_form_variables(self, human_task_id: str):
        response = requests.get(
            f"{self.camunda_config.base_url}/task/{human_task_id}/form-variables",
            timeout=self.timeout,
        )
        response.raise_for_status()
        instance_variables = response.json()

        try:
            rendered_form = self.get_human_task_rendered_form(human_task_id)
        except HTTPError as err:
            if err.response.status_code == 404:
                return {}  # no form variables if no rendered form!
            raise  # otherwise reraise

        # Extract form variables from the rendered form. Cannot use only camunda endpoint for form variables (broken)  # TODO link issue
        matches: List[str] = re.findall(
            r'<input[^\>]*cam-variable-name="(?P<name>[^"]*)"', rendered_form
        )  # returns a list of strings matching the 'name' group only
        form_variables = set(matches)

        return {k: v for k, v in instance_variables.items() if k in form_variables}

    def get_task_local_variables(self, external_task_execution_id: str):
        """
        Gets all local variables of an external task
        :param external_task_execution_id: The execution_id of the external task
        :return: Local variables
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/execution/{external_task_execution_id}/localVariables",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_global_variable(self, name: str, process_instance_id: str):
        """
        Retrieves the global variable for a given name
        :param name: The global variable name
        :param external_task
        :return: The value for a global variable
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/process-instance/{process_instance_id}/variables/{name}",
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()["value"]

    def get_instance_return_variables(self, process_instance_id: str):
        """
        Retrieves all workflow instance variables marked as workflow output
        :return: workflow output variables
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/process-instance/{process_instance_id}/variables",
            timeout=self.timeout,
        )
        response.raise_for_status()

        prefix = config["workflow_out"]["prefix"]

        return_variables = [
            {key: val} for key, val in response.json().items() if key.startswith(prefix)
        ]

        return return_variables

    def get_historic_process_instance_variables(self, process_instance_id: str):
        """
        Retrieves all workflow instance variables marked as workflow output from a completed workflow.
        :return: workflow output variables
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/history/variable-instance",
            params={
                "processInstanceId": process_instance_id,
                "variableNameLike": f'{config["workflow_out"]["prefix"]}%',
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()

    def get_task_execution_id(self, task_id: str):
        """
        Gets the execution id of an external task
        :param task_id: The task to get the execution id for
        :return: The execution id of an external task
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/external-task/{task_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()["executionId"]

    def get_human_task_rendered_form(self, task_id: str):
        """
        Retrieves the rendered form for a given human task
        :param task_id: The task id for a human task
        :return: Rendered HTML form
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/task/{task_id}/rendered-form",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def deploy(self, bpmn_location: Path) -> str:
        """
        Deploy a BPMN model to the Camunda Engine.
        Return the process definition id.
        """
        if bpmn_location is None:
            raise ValueError("No BPMN File specified!")
        with bpmn_location.open(mode="rb") as bpmn:
            response = requests.post(
                url=f"{self.camunda_config.base_url}/deployment/create",
                params={
                    "deployment-name": bpmn_location.name,
                    "enable-duplicate-filtering": "true",
                    "deployment-source": self.camunda_config.worker_id,  # TODO add path to deployment source? hashing?
                },
                files={bpmn_location.name: bpmn},
                timeout=self.timeout,
            )
            response.raise_for_status()

        deployment_id = response.json().get("id", None)
        process_def_response = requests.get(
            url=f"{self.camunda_config.base_url}/process-definition",
            params={
                "deploymentId": deployment_id,
            },
            timeout=self.timeout,
        )
        process_def_response.raise_for_status()
        deployed_process_defs = process_def_response.json()
        if len(deployed_process_defs) != 1:
            raise WorkflowDeploymentError(
                "The deployed BPMN file did not contain a process definition!"
            )

        process_definition_id = deployed_process_defs[0]["id"]
        return process_definition_id

    def create_instance(self, process_definition_id: str):
        """
        Create a workflow instance from the deployed BPMN model
        """
        response = requests.post(
            url=f"{self.camunda_config.base_url}/process-definition/{process_definition_id}/start",
            json={"variables": {}},
            timeout=self.timeout,
        )
        response.raise_for_status()

        process_instance_id = response.json()["id"]

        return process_instance_id

    def is_process_active(self, process_instance_id: str):
        """
        Checks if the process is still in the process list and thus active
        :return: Process status
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/process-instance/{process_instance_id}",
            timeout=self.timeout,
        )

        if response.status_code == 200:
            # TODO check what ended/suspended booleans in response mean
            return True
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return False

    def has_process_completed(self, process_instance_id: str):
        """
        Checks if the process instance was completed successully.
        :return: Process status
        """
        response = requests.get(
            f"{self.camunda_config.base_url}/history/process-instance/{process_instance_id}",
            timeout=self.timeout,
        )

        if response.status_code == 200:
            return response.json().get("state", "ERROR") == "COMPLETED"
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return False
