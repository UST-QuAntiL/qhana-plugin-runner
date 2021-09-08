from enum import Enum

import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class VariantEnum(Enum):
    negative_rotation = "Negative Rotation"
    destructive_interference = "Destructive Interference"
    state_preparation = "State Preparation"
    positive_correlation = "Positive Correlation"


class InputParameters:
    def __init__(self, entity_points_url: str, clusters_cnt: int, variant: VariantEnum):
        self.entity_points_url = entity_points_url
        self.clusters_cnt = clusters_cnt
        self.variant = variant


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Entity points URL",
            "description": "URL to a json file with the entity points.",
            "input_type": "text",
        },
    )
    clusters_cnt = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of clusters",
            "description": "Number of clusters that shall be found.",
            "input_type": "text",
        },
    )
    variant = EnumField(
        VariantEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Variant",
            "description": "Variant of quantum k-means that will be used.",
            "input_type": "select",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
