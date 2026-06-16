import marshmallow as ma
from marshmallow import post_load

from enum import Enum
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, FileUrl
from qhana_plugin_runner.api.extra_fields import EnumField

class RoutingOptions(Enum):
    WU_PALMER = "Wu-Palmer"
    ONE_HOT = "One-Hot"
    NUMMERIC_MAPPING = "Numeric Mapping"

class MerginPoint(Enum):
    TODO = "TODO"

class InputParameters:
    def __init__(
        self,
        entities_url: str,
        entities_metadata_url: str,
        taxonomies_zip_url: str,
        attributes: str,
        routing_options: RoutingOptions,
        merging_point: MerginPoint,
        taxonomy_checkbox: bool = False,
    ):
        self.entities_url = entities_url
        self.entities_metadata_url = entities_metadata_url
        self.taxonomies_zip_url = taxonomies_zip_url
        self.attributes = attributes
        self.routing_options = routing_options
        self.merging_point = merging_point
        self.taxonomy_checkbox = taxonomy_checkbox



class InputParametersSchema(FrontendFormBaseSchema):
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/list",
        data_content_types=["text/csv", "application/json"],
        metadata={
            "label": "Entities URL",
            "description": "URL to the entity list (e.g., subparts.csv).",
            "input_type": "text",
        },
    )
    entities_metadata_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/attribute-metadata",
        data_content_types="application/json",
        metadata={
            "label": "Entities Attribute Metadata URL",
            "description": "URL to a file with the attribute metadata for the entities.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "post",
        },
    )

    taxonomies_zip_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="graph/taxonomy",
        data_content_types="application/zip",
        metadata={
            "label": "Taxonomies URL",
            "description": "URL to zip file with taxonomies.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "pre",
        },
    )

    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Attributes",
            "description": "List of attributes for which the similarity shall be computed. Separated by newlines.",
            "input_type": "textarea",
        },
    )

    routing_options = EnumField(
        RoutingOptions,
        required=True,
        metadata={
            "label": "Routing Options",
            "description": "Select the routing option for similarity computation.",
            "input_type": "select",
        },
    )

    merging_point = EnumField(
        MerginPoint,
        required=True,
        metadata={
            "label": "Merging Point",
            "description": "Point in pipeline where different computation results should be merged. TODO.",
            "input_type": "select",
        },
    )

    taxonomy_checkbox = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Checkbox for testing (Does nothing)",
            "description": "TODO",
            "input_type": "checkbox",
        },
    )

