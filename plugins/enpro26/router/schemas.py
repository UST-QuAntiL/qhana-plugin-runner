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

from enum import Enum

import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


# Per-attribute pipeline options shown in the routing step.
PIPELINE_OPTIONS = ["Wu-Palmer", "One-Hot", "Mapping"]

PIPELINE_FIELD_PREFIX = "pipeline_"


# This Enum class is copied from the transformer plugin.
# Check the transformers plugin for updates
class TransformersEnum(Enum):
    linear_inverse = "Linear Inverse"
    exponential_inverse = "Exponential Inverse"
    gaussian_inverse = "Gaussian Inverse"
    polynomial_inverse = "Polynomial Inverse"
    square_inverse = "Square Inverse"


# This Enum class is copied from the aggregator plugin.
# Check the aggregator plugin for updates
class AggregatorsEnum(Enum):
    mean = "Mean"
    median = "Median"
    max = "Max"
    min = "Min"


# This Enum class is copied from the aggregator plugin.
# Check the aggregator plugin for updates
class MissingDataHandling(Enum):
    ignore = "ignore"
    mean = "mean"
    max = "max"


# This Enum class is copied from the mds plugin.
# Check the mds plugin for updates
class MetricEnum(Enum):
    metric_mds = "Metric MDS"
    nonmetric_mds = "Nonmetric MDS"


class InputParameters:
    def __init__(
        self,
        entities_url: str,
        entities_metadata_url: str,
        taxonomies_zip_url: str,
        root_is_part_of_hierarchy: bool,
        transformer: TransformersEnum,
        aggregator: AggregatorsEnum,
        missing_data_handling: MissingDataHandling,
        dimensions: int,
        metric: MetricEnum,
        n_init: int,
        max_iter: int,
    ):
        self.entities_url = entities_url
        self.entities_metadata_url = entities_metadata_url
        self.taxonomies_zip_url = taxonomies_zip_url
        self.root_is_part_of_hierarchy = root_is_part_of_hierarchy
        self.transformer = transformer
        self.aggregator = aggregator
        self.missing_data_handling = missing_data_handling
        self.dimensions = dimensions
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter


class InputParametersSchema(FrontendFormBaseSchema):
    # Base Inputs
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
        data_content_types=["application/json"],
        metadata={
            "label": "Entities Attribute Metadata URL",
            "description": "URL to a file with the attribute metadata for the entities.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "post",  # TODO: remove (?)
        },
    )
    taxonomies_zip_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="graph/taxonomy",
        data_content_types=["application/zip"],
        metadata={
            "label": "Taxonomies URL",
            "description": "URL to zip file with taxonomies.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "pre",
        },
    )

    # Pipeline Specific Inputs

    root_is_part_of_hierarchy = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={"label": "Root is part of hierarchy", "input_type": "checkbox"},
    )

    transformer = EnumField(
        TransformersEnum,
        required=True,
        metadata={
            "label": "Transformer",
            "description": "Transformer that shall be used to transform the similarities to distances.",
            "input_type": "select",
        },
    )
    aggregator = EnumField(
        AggregatorsEnum,
        required=True,
        metadata={
            "label": "Aggregator",
            "description": "Aggregator that shall be used to aggregate the attribute distances to a single distance value.",
            "input_type": "select",
        },
    )
    missing_data_handling = EnumField(
        MissingDataHandling,
        required=True,
        metadata={
            "label": "Missing data handling",
            "description": """Defines how a missing attribute distance should be handled.
- ignore: null values are removed and only the not null values are used for the aggregation
- mean: null values are replaced by the mean distance of the respective attribute
- max: null values are replaced by the maximum distance of the respective attribute""",
            "input_type": "select",
        },
    )
    dimensions = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Dimensions",
            "description": "Number of dimensions the output will have.",
            "input_type": "text",
        },
    )
    metric = EnumField(
        MetricEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Metric",
            "description": "Type of MDS that will be used.",
            "input_type": "select",
        },
    )
    n_init = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "SMACOF executions",
            "description": "Number of times SMACOF will be executed with different initial values.",
            "input_type": "text",
        },
    )
    max_iter = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "SMACOF max iterations",
            "description": "Maximum number of SMACOF iterations.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


class RoutingStepParametersSchema(FrontendFormBaseSchema):
    """Second step schema.

    The form renders one dropdown per taxonomy attribute with the field name
    ``pipeline_<attribute>``. The attributes are only known at runtime, so the
    fields are accepted dynamically instead of being declared statically.
    """

    @ma.validates_schema(pass_original=True)
    def validate_entries(self, data, original_data, **kwargs):
        errors = {}
        for key in original_data:
            if not key.startswith(PIPELINE_FIELD_PREFIX):
                errors[key] = [
                    f"Unexpected field '{key}', only "
                    f"'{PIPELINE_FIELD_PREFIX}<attribute>' is allowed."
                ]
                continue
            value = original_data[key]
            if value and value not in PIPELINE_OPTIONS:
                errors[key] = [f"'{value}' is not one of {PIPELINE_OPTIONS}."]
        if errors:
            raise ma.ValidationError(errors)

    @ma.post_load(pass_original=True)
    def add_dynamic_entries(self, data, original_data, **kwargs):
        # Each attribute maps to a single pipeline selection, so a flat
        # ``items()`` is sufficient for plain dicts and request MultiDicts alike.
        for key, value in original_data.items():
            if key.startswith(PIPELINE_FIELD_PREFIX):
                data[key] = value
        return data
