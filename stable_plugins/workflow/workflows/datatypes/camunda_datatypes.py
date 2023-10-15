import dataclasses
from typing import Optional, TypedDict, Literal, Union


class WorkflowIncident(TypedDict):
    id: str
    processDefinitionId: str
    processInstanceId: str
    executionId: Optional[str]
    incidentTimestamp: str
    incidentType: Union[Literal["failedJob", "failedExternalTask"], str]
    activityId: Optional[str]
    failedActivityId: Optional[str]
    causeIncidentId: Optional[str]
    rootCauseIncidentId: Optional[str]
    configuration: Optional[str]
    tenantId: Optional[str]
    incidentMessage: Optional[str]
    jobDefinitionId: Optional[str]
    annotation: Optional[str]


#
# CAMUNDA
# https://docs.camunda.org/manual/latest/reference/rest/
#
@dataclasses.dataclass
class ExternalTask:
    """
    process_instance_id: The id of the process instance the external task belongs to
    topic_name: The topic name of the external task
    """

    id: str
    execution_id: Optional[str]
    process_instance_id: str
    topic_name: str

    @classmethod
    def deserialize(cls, serialized):
        return cls(
            id=serialized["id"],
            execution_id=serialized["executionId"],
            process_instance_id=serialized["processInstanceId"],
            topic_name=serialized["topicName"],
        )


@dataclasses.dataclass
class HumanTask:
    """
    TODO: Docs for HumanTask
    """

    id: str
    name: str
    execution_id: str
    assignee: Optional[str]
    delegation_state: str
    process_instance_id: str
    task_definition_key: str

    @classmethod
    def deserialize(cls, serialized):
        return cls(
            id=serialized["id"],
            name=serialized["name"],
            execution_id=serialized["executionId"],
            assignee=None if serialized["assignee"] is None else serialized["assignee"],
            delegation_state=serialized["delegationState"],
            process_instance_id=serialized["processInstanceId"],
            task_definition_key=serialized["taskDefinitionKey"],
        )
