# Copyright 2026 QHAna plugin runner contributors.
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

"""
This module defines enums for allowed data_types and content_types used in plugin metadata.
Types needs to be added manually to these Enums.

The Enums are organized into:
- Data types, that follow a specific format (namespace/name)
- Data types, that do not follow a specific format
- Content types, that should be RFC-compliant.
"""

from enum import Enum


class AllowedDataTypesWithFormat(Enum):
    """
    Allowed data types of the following format: namespace/name.\\
    Types, that do not follow this format can be find in :func:`AllowedDataTypesNoFormat`.
    """

    # Standard data types of format namespace/name
    CUSTOM_CLUSTERS = "custom/clusters"
    CUSTOM_HELLO_WORLD_OUTPUT = "custom/hello-world-output"
    CUSTOM_KERNEL_MATRIX = "custom/kernel-matrix"
    CUSTOM_PCA_METADATA = "custom/pca-metadata"
    CUSTOM_PLOT = "custom/plot"
    ENTITY_WILDCARD = "entity/*"
    ENTITY_ATTRIBUTE_METADATA = "entity/attribute-metadata"
    ENTITY_LABEL = "entity/label"
    ENTITY_LIST = "entity/list"
    ENTITY_MATRIX = "entity/matrix"
    ENTITY_SHAPED_VECTOR = "entity/shaped_vector"
    ENTITY_VECTOR = "entity/vector"
    EXECUTABLE_CIRCUIT = "executable/circuit"
    GRAPH_TAXONOMY = "graph/taxonomy"
    IMAGE_HTML = "image/html"
    PROVENANCE_EXECUTION_OPTIONS = "provenance/execution-options"
    PROVENANCE_TRACE = "provenance/trace"
    RELATION_ATTRIBUTE_DISTANCES = "relation/attribute-distances"
    RELATION_ATTRIBUTE_SIMILARITIES = "relation/attribute-similarities"
    RELATION_ELEMENT_SIMILARITIES = "relation/element-similarities"
    RELATION_ENTITY_DISTANCES = "relation/entity-distances"
    TABLE_HTML = "table/html"


class AllowedDataTypesNoFormat(Enum):
    """
    Allowed data types that do not follow a specific format.\\
    Data types that use the format (namespace/name) can be found in :func:`AllowedDataTypesWithFormat`.
    """

    WILDCARD = "*"
    CIRCUIT = "circuit"
    PLOT = "plot"
    QNN_WEIGHTS = "qnn-weights"
    REPRESENTATIVE_CIRCUIT = "representative-circuit"
    TXT = "txt"
    VQC_METADATA = "vqc-metadata"


class AllowedContentTypes(Enum):
    """
    Allowed content types (mimetypes).
    They should be RFC-compliant.
    """

    WILDCARD = "*"
    AUDIO_MIDI = "audio/midi"
    AUDIO_X_MIDI = "audio/x-midi"
    APPLICATION_X_LINES_PLUS_JSON = "application/X-lines+json"
    APPLICATION_JSON = "application/json"
    APPLICATION_OCTET_STREAM = "application/octet-stream"
    APPLICATION_QASM = "application/qasm"
    APPLICATION_VND_RECORDARE_MUSICXML_PLUS_XML = "application/vnd.recordare.musicxml+xml"
    APPLICATION_XML = "application/xml"
    APPLICATION_ZIP = "application/zip"
    IMAGE_SVG_PLUS_XML = "image/svg+xml"
    TEXT_CSV = "text/csv"
    TEXT_HTML = "text/html"
    TEXT_PLAIN = "text/plain"
    TEXT_XML = "text/xml"
    TEXT_X_QASM = "text/x-qasm"


def is_valid_data_type(value: str) -> bool:
    """Check if a data_type value is in allowed lists."""
    return any(e.value == value for e in AllowedDataTypesWithFormat) or any(
        e.value == value for e in AllowedDataTypesNoFormat
    )


def is_valid_content_type(value: str) -> bool:
    """Check if a content_type value is in allowed lists."""
    return any(e.value == value for e in AllowedContentTypes)
