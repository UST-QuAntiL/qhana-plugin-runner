# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import OrderedDict
from enum import Enum
from typing import Dict, Iterable, Literal

import marshmallow as ma
from marshmallow import INCLUDE
from marshmallow.validate import OneOf
from typing_extensions import Required, TypedDict

from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema, MaBaseSchema


class ConnectorKey(Enum):
    # set data commands
    NAME = "name"
    DESCRIPTION = "description"
    BASE_URL = "base-url"
    OPENAPI_SPEC = "openapi-spec"
    ENDPOINT_URL = "endpoint-url"
    VARIABLES = "variables"
    REQUEST_HEADERS = "request-headers"
    REQUEST_BODY = "request-body"
    REQUEST_FILES = "request-files"
    RESPONSE_HANDLING = "response-handling"
    RESPONSE_MAPPING = "response-mapping"

    # general commands
    DEPLOY = "deploy"
    UNDEPLOY = "undeploy"
    CANCEL = "cancel"


class VariableType(Enum):
    STRING = "string"
    TEXT = "text"
    NUMBER = "number"
    INTEGER = "integer"
    DATA = "data"


class WelcomeParametersSchema(FrontendFormBaseSchema):
    api_name = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "REST API Name",
            "description": "The name of the REST API.",
            # "input_type": "url",
        },
    )


class ConnectorUpdateSchema(MaBaseSchema):
    key = EnumField(ConnectorKey, required=True, use_value=True)
    value = ma.fields.String(required=True)
    next_step = ma.fields.String(required=False, missing="")


class ConnectorVariableSchema(MaBaseSchema):
    name = ma.fields.String(required=True)
    type = ma.fields.String(
        required=True, validate=OneOf({e.value for e in VariableType})
    )
    default_value = ma.fields.String(required=False, allow_none=True)
    title = ma.fields.String(required=False, allow_none=True)
    description = ma.fields.String(required=False, allow_none=True)
    data_type = ma.fields.String(required=False, allow_none=True)
    content_type = ma.fields.String(required=False, allow_none=True)
    required = ma.fields.Boolean(required=False, allow_none=True)
    load_content = ma.fields.Boolean(required=False, allow_none=True)


class ConnectorVariable(TypedDict, total=False):
    name: Required[str]
    type: Required[Literal["string", "text", "number", "integer", "data"]]
    default_value: str
    title: str
    description: str
    data_type: str
    content_type: str
    required: bool
    load_content: bool


class RequestFileDescriptorSchema(MaBaseSchema):
    source = ma.fields.String(required=True)
    name = ma.fields.String(required=False, allow_none=True)
    content_type = ma.fields.String(required=False, allow_none=True)


class ResponseOutput(TypedDict, total=False):
    name: Required[str]
    data_type: Required[str]
    content_type: Required[str]
    data: Required[str]
    dereference_url: bool


class ResponseOutputSchema(MaBaseSchema):
    name = ma.fields.String(required=True)
    data_type = ma.fields.String(required=True)
    content_type = ma.fields.String(required=True)
    data = ma.fields.String(required=True)
    dereference_url = ma.fields.Boolean(required=False)


class ConnectorSchema(MaBaseSchema):
    name = ma.fields.String(dump_only=True)
    version = ma.fields.Integer(dump_only=True, dump_default=0)
    description = ma.fields.String(dump_default="")
    is_deployed = ma.fields.Boolean(dump_only=True, dump_default=False)
    is_loading = ma.fields.Boolean(dump_only=True, dump_default=False)
    next_step = ma.fields.String(dump_default="")
    base_url = ma.fields.Url(dump_default="")
    openapi_spec_url = ma.fields.Url(dump_default="")
    endpoint_url = ma.fields.Url(dump_default="")
    variables = ma.fields.List(
        ma.fields.Nested(ConnectorVariableSchema), dump_default=tuple()
    )
    request_headers = ma.fields.String(dump_default="")
    request_body = ma.fields.String(dump_default="")
    request_files = ma.fields.List(
        ma.fields.Nested(RequestFileDescriptorSchema), dump_default=tuple()
    )
    response_handling = ma.fields.String(dump_default="")
    response_mapping = ma.fields.List(
        ma.fields.Nested(ResponseOutputSchema), dump_default=tuple()
    )


class ConnectorVariablesInputSchema(FrontendFormBaseSchema):
    def __init__(self, variables: Iterable[ConnectorVariable] = tuple(), **kwargs):
        if "unknown" not in kwargs:
            kwargs["unknown"] = INCLUDE
        super().__init__(**kwargs)

        inputs: Dict = {}

        # TODO: Default values
        # TODO: selects

        for var in variables:
            type_ = var["type"]
            if type_ == "text" or type_ == "string":
                inputs[var["name"]] = ma.fields.String(
                    required=True,
                    allow_none=False,
                    data_key=var["name"],
                    metadata={
                        "label": var.get("title", var["name"]),
                        "description": var.get("description", ""),
                        "input_type": "textarea" if type_ == "text" else "text",
                    },
                )
            elif type_ == "number":
                inputs[var["name"]] = ma.fields.Float(
                    required=True,
                    allow_none=False,
                    data_key=var["name"],
                    metadata={
                        "label": var.get("title", var["name"]),
                        "description": var.get("description", ""),
                    },
                )
            elif type_ == "integer":
                inputs[var["name"]] = ma.fields.Integer(
                    required=True,
                    allow_none=False,
                    data_key=var["name"],
                    metadata={
                        "label": var.get("title", var["name"]),
                        "description": var.get("description", ""),
                    },
                )
            elif type == "data":
                inputs[var["name"]] = FileUrl(
                    required=True,
                    allow_none=False,
                    data_key=var["name"],
                    data_input_type=var.get("data_type", "*"),
                    data_content_types=var.get("content_type", "*"),
                    metadata={
                        "label": var.get("title", var["name"]),
                        "description": var.get("description", ""),
                    },
                )

        self.fields = OrderedDict(inputs)
