from typing import Optional


class CamundaError(Exception):
    """Base exception for errors during interaction with camunda."""

    def __init__(self, *args: object, message: Optional[str] = None) -> None:
        self.message = message
        super().__init__(message, *args)


class CamundaServerError(CamundaError):
    """Camunda returned an error code indicating a server error."""


class CamundaClientError(CamundaError):
    """Camunda returned an error code indicating a client error."""


class WorkflowDeploymentError(CamundaError):
    """The workflow could not be deployed."""


class WorkflowNotFoundError(CamundaError):
    """The requested workflow was not found."""


class WorkflowStoppedError(CamundaError):
    """The workflow was stopped unexpectedly."""


class WorkflowTaskError(CamundaError):
    """Base error class for errors while executing an external task."""


class InvocationError(WorkflowTaskError):
    """Error for when the task failed because it could not be invoked for any reason."""


class PluginNotFoundError(InvocationError):
    """Error for when the task failed because the plugin could not be found."""


class StepNotFoundError(InvocationError):
    """Error for when the task failed because a plugin step could not be found."""


class BadTaskDefinitionError(InvocationError):
    """Error for when the task failed because the task definition in the workflow file contained errors."""


class BadInputsError(InvocationError):
    """Error for when the task failed because the inputs were incomplete or malformed."""


class ExecutionError(WorkflowTaskError):
    """Error for when a plugin was invoked but execution fails."""


class ResultError(ExecutionError):
    """Error when the execution failed because of an issue with the result resource."""


class PluginFailureError(ExecutionError):
    """Error for when a plugin execution ends with a 'FAILURE' result."""
