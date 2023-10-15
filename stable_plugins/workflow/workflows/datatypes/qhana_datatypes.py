import dataclasses
from typing import List


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


@dataclasses.dataclass
class QhanaInput:
    """ """

    content_type: List[str]
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
