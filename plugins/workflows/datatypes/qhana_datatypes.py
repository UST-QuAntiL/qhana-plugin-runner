import dataclasses
from typing import List

from dataclasses_json import (
    dataclass_json,  # TODO maybe remove dependency after refactor
)

from ..datatypes.camunda_datatypes import ExternalTask


#
# QHAna
# https://<endpoint>/rapidoc
#
@dataclasses.dataclass
class QhanaPlugin:
    """
    api_endpoint: Endpoint of the repository containing plugins
    api_root: Plugin endpoint
    identifier: Identifier of the plugin
    name: Plugin name
    version: Plugin version
    """

    api_endpoint: str
    api_root: str
    process_endpoint: str
    identifier: str
    name: str
    version: str

    @classmethod
    def deserialize(cls, serialized, endpoint, process_endpoint):
        return cls(
            api_root=serialized["apiRoot"].rstrip("/"),
            identifier=serialized["identifier"],
            name=serialized["name"],
            version=serialized["version"],
            api_endpoint=endpoint.rstrip("/"),
            process_endpoint=process_endpoint,
        )


@dataclass_json
@dataclasses.dataclass
class QhanaTask:
    """
    external_task: The corresponding external camunda task
    plugin: QhanaPlugin executing the QhanaTask
    name: Name of the QhanaTask
    status: QhanaTask status
    id: QhanaTask identifier
    """

    # TODO: Cannot store inputs until Human Tasks to collect inputs are also done in QHAna
    #  (Human Tasks in Camunda are missing href, contentType etc.)
    external_task: ExternalTask
    plugin: QhanaPlugin
    status: str
    id: str

    @classmethod
    def deserialize(cls, serialized, db_id, external_task, plugin):
        return cls(
            status=serialized["status"],
            id=db_id,
            external_task=external_task,
            plugin=plugin,
        )


@dataclasses.dataclass
class QhanaInput:
    """ """

    content_type: list[str]
    data_type: str
    required: bool

    @classmethod
    def deserialize(cls, serialized):
        return cls(
            content_type=serialized["contentType"],
            data_type=serialized["dataType"],
            required=serialized["required"],
        )


@dataclasses.dataclass
class QhanaOutput:
    """
    content_type: The media type (mime type) of the output data (e.g. application/json)
    href: URL of the concrete output data
    name: The (default) name of the output data
    output_type: The type of the output (e.g. distance-matrix)
    """

    content_type: str
    data_type: str
    href: str
    name: str

    @classmethod
    def deserialize(cls, serialized):
        return cls(
            content_type=serialized["contentType"],
            data_type=serialized["dataType"],
            href=serialized["href"],
            name=serialized["name"],
        )


@dataclasses.dataclass
class QhanaResult:
    """
    qhana_task: The QhanaTask which produced the result
    output_list: List containing all outputs
    """

    qhana_task: QhanaTask
    output_list: List[QhanaOutput]
