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
from typing import List, Optional

import marshmallow as ma

from qhana_plugin_runner.api.util import MaBaseSchema


class FileMetadataSchema(MaBaseSchema):
    file_type = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={"description": "The type of the file (e.g. distance-matrix)."},
    )
    mimetype = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={"description": "The mime type of the file (e.g. application/json)."},
    )
    filename = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={"description": "The (default) file name."},
    )


class ConcreteFileMetadataSchema(FileMetadataSchema):
    href = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={"description": "URL of the concrete file."},
    )


@dataclass
class FileMetadata:
    file_type: str
    mimetype: str
    filename: Optional[str] = None
    href: Optional[str] = None


class ProcessingResourceMetadataSchema(MaBaseSchema):
    href = ma.fields.Url(
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={"description": "URL of the processing resource of the plugin."},
    )
    ui_href = ma.fields.Url(
        required=False,
        allow_none=False,
        dump_only=True,
        metadata={
            "description": "URL of the micro frontend for the processing resource of the plugin."
        },
    )
    file_inputs = ma.fields.List(
        ma.fields.List(ma.fields.Nested(FileMetadataSchema())),
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={"description": "A list of possible file input sets."},
    )
    file_outputs = ma.fields.List(  # for future use
        ma.fields.List(ma.fields.Nested(FileMetadataSchema())),
        required=False,
        allow_none=False,
        dump_only=True,
        missing=[],
        metadata={
            "description": "A list of guaranteed output files (independent from which input file types were provided and which parameters are set)."
        },
    )


@dataclass
class ProcessingResourceMetadata:
    href: str
    ui_href: Optional[str]
    file_inputs: List[List[FileMetadata]] = field(default_factory=list)
    file_outputs: List[FileMetadata] = field(default_factory=list)


class PluginMetadataSchema(MaBaseSchema):
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
    identifier = ma.fields.String(
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={
            "description": "Unique identifier of the plugin (derived from name and version)."
        },
    )
    root_href = ma.fields.Url(
        required=False,
        allow_none=False,
        dump_only=True,
        metadata={"description": "URL of the api root of the plugin."},
    )
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
    plugin_type = ma.fields.String(
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={
            "description": "Type of the plugin (e.g. data-loader, data-processor, data-visualizer)."
        },
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


class DataProcessingPluginMetadataSchema(MaBaseSchema):
    processing_resource_metadata = ma.fields.Nested(
        ProcessingResourceMetadataSchema(),
        required=True,
        allow_none=False,
        dump_only=True,
        metadata={
            "description": "Metadata for the data processing resource of the plugin."
        },
    )


@dataclass
class PluginMetadata:
    name: str
    version: str
    identifier: str
    root_href: str
    title: str
    description: str
    plugin_type: str
    tags: List[str] = field(default_factory=list)
    processing_resource_metadata: Optional[ProcessingResourceMetadata] = None
