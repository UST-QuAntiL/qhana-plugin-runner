"""
Auto-generated module for QHAna plugin data types and content types.

This module defines enums for allowed data_types and content_types used in plugin metadata.
Types are extracted from all plugins in plugins/ and stable_plugins/.

Types are organized into:
- Standard types: Follow namespace/name format or wildcards
- Legacy types: Non-standard but used in plugins (backward compatibility)
- Suspicious types: Technically allowed but semantically incorrect per standards (TODO: fix in plugins)
"""

from enum import Enum


class AllowedDataTypesWithFormat(Enum):
    """
    Allowed data types of the following format: namespace/name.\\
    Data types, that are allowed, but do not follow this format
    Also contains some data types, that DO NOT follow this format and *.
    """

    # Standard data types of format namespace/name
    CUSTOM_ATTRIBUTE_DISTANCES = "custom/attribute-distances"
    CUSTOM_ATTRIBUTE_SIMILARITIES = "custom/attribute-similarities"
    CUSTOM_CLUSTERS = "custom/clusters"
    CUSTOM_ELEMENT_SIMILARITIES = "custom/element-similarities"
    CUSTOM_ENTITY_DISTANCES = "custom/entity-distances"
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
    TABLE_HTML = "table/html"


class AllowedDataTypesNoFormat(Enum):
    # Other data types
    WILDCARD = "*"
    CIRCUIT = "circuit"
    PLOT = "plot"
    QNN_WEIGHTS = "qnn-weights"
    REPRESENTATIVE_CIRCUIT = "representative-circuit"
    TXT = "txt"
    VQC_METADATA = "vqc-metadata"


class AllowedContentTypes(Enum):
    """Allowed content types (mimetypes): RFC-compliant."""

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
