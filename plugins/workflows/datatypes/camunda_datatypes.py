import dataclasses


#
# CAMUNDA
# https://docs.camunda.org/manual/latest/reference/rest/
#
from typing import Optional


@dataclasses.dataclass
class ExternalTask:
    """
    process_instance_id: The id of the process instance the external task belongs to
    topic_name: The topic name of the external task
    """
    id: str
    execution_id: str
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
    execution_id: str
    assignee: Optional[str]
    delegation_state: str
    process_instance_id: str

    @classmethod
    def deserialize(cls, serialized):
        return cls(
            id=serialized["id"],
            execution_id=serialized["executionId"],
            assignee=None if serialized["assignee"] is None else serialized["assignee"],
            delegation_state=serialized["delegationState"],
            process_instance_id=serialized["processInstanceId"],
        )


@dataclasses.dataclass
class Deployment:
    """
    id: The id of the deployment
    process_definition_id: The id of the process definition
    """
    id: str
    process_definition_id: str

    @classmethod
    def deserialize(cls, serialized):
        return cls(
            id=serialized["id"],
            process_definition_id=list(serialized["deployedProcessDefinitions"].keys())[0]
        )


@dataclasses.dataclass
class ProcessInstance:
    """
    id: The identifier of the process instance
    """
    id: str

    @classmethod
    def deserialize(cls, serialized):
        return cls(
            id=serialized["id"]
        )