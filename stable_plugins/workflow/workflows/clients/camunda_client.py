import datetime
import logging
import re
from typing import List, Mapping, Optional, Sequence
from xml.etree import ElementTree

import requests
from requests.exceptions import HTTPError
from werkzeug.utils import secure_filename

from qhana_plugin_runner.requests import open_url_as_file_like

from ..config import WorkflowPluginConfig, separate_prefixes
from ..datatypes.camunda_datatypes import ExternalTask, HumanTask, WorkflowIncident
from ..exceptions import WorkflowDeploymentError

logger = logging.getLogger(__name__)


def extract_id_and_name(bpmn_url: str):
    with open_url_as_file_like(bpmn_url) as (filename, bpmn, _):
        id_ = "unknown"
        name = filename
        for _event, node in ElementTree.iterparse(bpmn, ["start"]):
            if node.tag == "{http://www.omg.org/spec/BPMN/20100524/MODEL}definitions":
                continue
            if node.tag == "{http://www.omg.org/spec/BPMN/20100524/MODEL}process":
                name = node.attrib.get("name", name)
                id_ = node.attrib["id"]
            elif node.tag.endswith("process"):
                name = node.attrib.get("name", name)
                id_ = node.attrib.get("id", id_)
            break
        return id_, name


class CamundaManagementClient:
    # http://localhost:8080/swaggerui/

    def __init__(self, config: WorkflowPluginConfig, timeout: float | None = None):
        self.camunda_endpoint = config["camunda_base_url"].rstrip("/")
        self.worker_id = config["worker_id"]
        self.timeout = timeout if timeout else config["request_timeout"]

    def deploy_bpmn(self, bpmn_url: str) -> str:
        """
        Deploy a BPMN model to the Camunda Engine.
        Return the process definition id.
        """
        if not bpmn_url:
            raise ValueError("No BPMN File specified!")
        id_, name = extract_id_and_name(bpmn_url)
        with open_url_as_file_like(bpmn_url) as (filename, bpmn, content_type):
            sec_file_name = secure_filename(filename + ".bpmn")
            file_ = (
                (sec_file_name, bpmn, content_type)
                if content_type
                else (sec_file_name, bpmn)
            )
            response = requests.post(
                url=f"{self.camunda_endpoint}/deployment/create",
                params={
                    "deployment-name": id_,
                    "enable-duplicate-filtering": "true",
                    "deployment-source": self.worker_id,
                    # TODO add path to deployment source? hashing?
                },
                files={id_: file_},
                timeout=self.timeout,
            )
            response.raise_for_status()

        deployment_id = response.json().get("id", None)
        process_def_response = requests.get(
            url=f"{self.camunda_endpoint}/process-definition",
            params={
                "deploymentId": deployment_id,
            },
            timeout=self.timeout,
        )
        process_def_response.raise_for_status()
        deployed_process_defs = process_def_response.json()
        if len(deployed_process_defs) != 1:
            raise WorkflowDeploymentError(
                f"The deployed BPMN file did not contain a process definition! (deploymentId: {deployment_id})"
            )

        process_definition_id = deployed_process_defs[0]["id"]
        return process_definition_id

    def get_process_definitions(self):
        response = requests.get(
            url=f"{self.camunda_endpoint}/process-definition",
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()

    def get_process_definition(self, definition_id: str):
        response = requests.get(
            url=f"{self.camunda_endpoint}/process-definition/{definition_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()

    def get_process_definition_xml(self, definition_id: str):
        response = requests.get(
            url=f"{self.camunda_endpoint}/process-definition/{definition_id}/xml",
            timeout=self.timeout,
        )
        response.raise_for_status()

        return response.json()["bpmn20Xml"]

    def get_workflow_start_form_variables(self, definition_id: str):
        response = requests.get(
            url=f"{self.camunda_endpoint}/process-definition/{definition_id}/form-variables",
            timeout=self.timeout,
        )
        response.raise_for_status()
        start_variables = response.json()

        try:
            rendered_form = self.get_workflow_start_rendered_form(definition_id)
        except HTTPError as err:
            if err.response.status_code == 404:
                return {}  # no form variables if no rendered form!
            raise  # otherwise reraise

        # Extract form variables from the rendered form. Cannot use only camunda endpoint for form variables (broken)  # TODO link issue
        matches: List[str] = re.findall(
            r'<input[^\>]*cam-variable-name="(?P<name>[^"]*)"', rendered_form
        )  # returns a list of strings matching the 'name' group only
        form_variables = set(matches)

        return {k: v for k, v in start_variables.items() if k in form_variables}

    def get_workflow_start_rendered_form(self, definition_id: str):
        response = requests.get(
            url=f"{self.camunda_endpoint}/process-definition/{definition_id}/rendered-form",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def start_workflow(self, definition_id: str, form_inputs: Mapping):
        response = requests.post(
            url=f"{self.camunda_endpoint}/process-definition/{definition_id}/start",
            json={"variables": form_inputs},
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()


class CamundaClient:
    """
    Handles setup for deployments, process instances and listeners.
    Removes deployment after instance has finished
    """

    def __init__(self, config: WorkflowPluginConfig, timeout: float | None = None):
        self.base_url = config["camunda_base_url"].rstrip("/")
        self.worker_id = config["worker_id"]
        self.timeout = timeout if timeout else config["request_timeout"]
        self.workflow_conf = config["workflow_conf"]

    def lock(self, external_task_id: str):
        """
        Locks an external task so that the worker can use it
        :param task: The task to be locked
        """
        # TODO: Instead of just setting a high lock duration, automatically re-lock when expired
        response = requests.post(
            f"{self.base_url}/external-task/{external_task_id}/lock",
            json={"workerId": self.worker_id, "lockDuration": "999999999"},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def get_locked_external_tasks_count(self) -> int:
        response = requests.get(
            f"{self.base_url}/external-task/count",
            params={
                "workerId": self.worker_id,
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
        params = {"active": "true", "notLocked": "true", "withRetriesLeft": "true"}
        if limit is not None:
            if limit < 1:
                raise ValueError("The limit must not be smaller than 1!")
            params["maxResults"] = str(limit)
        response = requests.get(
            f"{self.base_url}/external-task",
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
            f"{self.base_url}/external-task/{external_task_id}/unlock",
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
            f"{self.base_url}/external-task/{external_task_id}",
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
        output_variables = {
            "output": {
                "value": "Exception thrown, see error code and error message!",
                "type": "String",
            }
        }
        response = requests.post(
            url=f"{self.base_url}/external-task/{external_task_id}/bpmnError",
            json={
                "workerId": self.worker_id,
                "errorCode": error_code,
                "errorMessage": error_message,
                "variables": output_variables,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

    def external_task_report_failure(
        self, external_task_id: str, error_code: str, error_message: str
    ):
        """
        Notifies camunda of the task failure.
        :param external_task_id: The id of the failed task
        :param error_code: Code for exception
        :param error_message: Description of the exception
        """
        response = requests.post(
            url=f"{self.base_url}/external-task/{external_task_id}/failure",
            json={
                "workerId": self.worker_id,
                "errorMessage": f"{error_code}: {error_message}",
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
            f"{self.base_url}/external-task/{external_task_id}/complete",
            json={"workerId": self.worker_id, "variables": result},
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
            f"{self.base_url}/task/{human_task_id}/complete",
            json={"workerId": self.worker_id, "variables": result},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def get_workflow_incidents(
        self, process_instance_id: str
    ) -> Sequence[WorkflowIncident]:
        """Get all unresolved incidents of a given process instance."""
        response = requests.get(
            f"{self.base_url}/incident",
            params={
                "processInstanceId": process_instance_id,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_workflow_incident(self, incident_id: str) -> WorkflowIncident:
        """Get a specific incident."""
        response = requests.get(
            f"{self.base_url}/incident/{incident_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def resolve_workflow_incident(self, incident_id: str) -> None:
        """Resolve a specific incident (only works for some incidents!)."""
        response = requests.delete(
            f"{self.base_url}/incident/{incident_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()

    def get_human_tasks(self, process_instance_id: str) -> Sequence[HumanTask]:
        """Get all active human tasks of a given process instance."""
        response = requests.get(
            f"{self.base_url}/task",
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
            f"{self.base_url}/task/{human_task_id}/form-variables",
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
            f"{self.base_url}/execution/{external_task_execution_id}/localVariables",
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
            f"{self.base_url}/process-instance/{process_instance_id}/variables/{name}",
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
            f"{self.base_url}/process-instance/{process_instance_id}/variables",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return_prefix = self.workflow_conf["return_variable_prefix"]

        variables = (
            (*separate_prefixes(key, self.workflow_conf), val)
            for key, val in response.json().items()
        )

        return [
            {name: val} for name, prefixes, val in variables if return_prefix in prefixes
        ]

    def get_historic_process_instance_variables(self, process_instance_id: str):
        """
        Retrieves all workflow instance variables marked as workflow output from a completed workflow.
        :return: workflow output variables
        """
        return_prefix = self.workflow_conf["return_variable_prefix"]

        response = requests.get(
            f"{self.base_url}/history/variable-instance",
            params={
                "processInstanceId": process_instance_id,
                "variableNameLike": f'%{return_prefix}{self.workflow_conf["prefix_separator"]}%',
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        variables = (
            (*separate_prefixes(var["name"], self.workflow_conf), var)
            for var in response.json()
        )

        return {
            name: val for name, prefixes, val in variables if return_prefix in prefixes
        }

    def get_task_execution_id(self, task_id: str):
        """
        Gets the execution id of an external task
        :param task_id: The task to get the execution id for
        :return: The execution id of an external task
        """
        response = requests.get(
            f"{self.base_url}/external-task/{task_id}",
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
            f"{self.base_url}/task/{task_id}/rendered-form",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def is_process_active(self, process_instance_id: str):
        """
        Checks if the process is still in the process list and thus active
        :return: Process status
        """
        response = requests.get(
            f"{self.base_url}/process-instance/{process_instance_id}",
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
            f"{self.base_url}/history/process-instance/{process_instance_id}",
            timeout=self.timeout,
        )

        if response.status_code == 200:
            return response.json().get("state", "ERROR") == "COMPLETED"
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return False

    def cancel_running_workflow(self, process_instance_id: str) -> None:
        """Cancel a running workflow (e.g. if an unrecoverable incident occurred)."""
        response = requests.delete(
            f"{self.base_url}/process-instance/{process_instance_id}",
            params={"skipIoMappings": "true", "failIfNotExists": "false"},
            timeout=self.timeout,
        )
        response.raise_for_status()
