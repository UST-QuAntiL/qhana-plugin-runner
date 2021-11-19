# Copyright 2021 QHAna plugin runner contributors.
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

"""Module containing schemas to be used by plugins."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

import marshmallow as ma

from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import MaBaseSchema


@dataclass
class DataMetadata:
    data_type: str
    content_type: List[str]
    required: bool


class DataMetadataSchema(MaBaseSchema):
    data_type = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={"description": "The type of the output (e.g. distance-matrix)."},
    )
    content_type = ma.fields.List(
        ma.fields.String,
        required=True,
        allow_none=False,
        metadata={
            "description": "The media type (mime type) of the output data (e.g. application/json)."
        },
    )
    required = ma.fields.Boolean(
        required=True,
        allow_none=False,
        metadata={"description": "If the data is required or not."},
    )


class ProgressMetadataSchema(MaBaseSchema):
    value = ma.fields.Integer(
        required=True, allow_none=False, metadata={"description": "The progress value."}
    )
    start = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={"description": "The progress start value."},
    )
    target = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={"description": "The progress target value."},
    )
    unit = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "description": "The progress unit."
        },  # TODO: only allow limited choice, e.g., %
    )


class StepMetadataSchema(MaBaseSchema):
    href = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={"description": "The URL of the REST entry point resource."},
    )
    uiHref = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "description": "The URL of the micro frontend that corresponds to the REST entry point resource."
        },
    )
    stepId = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={"description": 'ID of step, e.g., ``"step1"`` or ``"step1.step2b"``.'},
    )
    cleared = ma.fields.Boolean(
        required=False,
        allow_none=True,
        metadata={
            "description": "``false`` if step is awaiting input, only last step in list can be marked as ``false``."
        },
    )


@dataclass
class EntryPoint:
    href: str
    ui_href: str
    data_input: List[DataMetadata] = field(default_factory=list)
    data_output: List[DataMetadata] = field(default_factory=list)


@dataclass
class ProgressMetadata:
    value: int
    start: int
    target: int
    unit: str


@dataclass
class StepMetadata:
    href: str
    uiHref: str
    stepId: str
    cleared: bool = False


class EntryPointSchema(MaBaseSchema):
    href = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={"description": "The URL of the REST entry point resource."},
    )
    ui_href = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={
            "description": "The URL of the micro frontend that corresponds to the REST entry point resource."
        },
    )
    data_input = ma.fields.List(
        ma.fields.Nested(
            DataMetadataSchema,
            required=True,
            allow_none=False,
            metadata={"description": "A list of possible data inputs."},
        )
    )
    data_output = ma.fields.List(
        ma.fields.Nested(
            DataMetadataSchema,
            required=True,
            allow_none=False,
            metadata={"description": "A list of possible data outputs."},
        )
    )


class PluginType(Enum):
    simple = "simple"
    complex = "complex"


@dataclass
class PluginMetadata:
    title: str
    description: str
    name: str
    version: str
    type: PluginType
    entry_point: EntryPoint
    tags: List[str] = field(default_factory=list)


class PluginMetadataSchema(MaBaseSchema):
    title = ma.fields.String(
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={"description": "Human readable plugin title."},
    )
    description = ma.fields.String(
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={"description": "Human readable plugin description."},
    )
    name = ma.fields.String(
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={"description": "Unique name of the plugin."},
    )
    version = ma.fields.String(
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={"description": "Version of the plugin."},
    )
    type = EnumField(
        PluginType,
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={"description": "Type of the plugin"},
    )
    entry_point = ma.fields.Nested(
        EntryPointSchema,
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={"description": "The entry point of the plugin"},
    )
    tags = ma.fields.List(
        ma.fields.String(),
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={
            "description": "A list of tags describing the plugin (e.g. classical-algorithm, quantum-algorithm, hybrid-algorithm)."
        },
    )
