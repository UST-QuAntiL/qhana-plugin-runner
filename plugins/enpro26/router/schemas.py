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
import textwrap

import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema

NONE_PLUGIN = "none"
WU_PALMER_PLUGIN = "wu_palmer"
MAPPING_PLUGIN = "mapping"
ONE_HOT_PLUGIN = "one_hot"
TRANSFORMERS_PLUGIN = "transformers"
AGGREGATOR_PLUGIN = "aggregator"
MDS_PLUGIN = "mds"
VECTOR_CONCAT_PLUGIN = "vector_concat"

FINALIZE_STEP = "finalize"

# Names of the plugins invoked by the routing pipeline. The runner serves
# plugin metadata at ``/plugins/<name>/`` and redirects a bare name to the
# newest installed version. The route handler turns these into external
# metadata urls (see routes.py), from which the process endpoint is resolved
# through ``get_plugin_endpoint``.
PIPELINE_PLUGINS = {
    WU_PALMER_PLUGIN: "wu-palmer",
    MAPPING_PLUGIN: "mapping-distances",
    TRANSFORMERS_PLUGIN: "element_sim-to-element_dist-transformers",
    AGGREGATOR_PLUGIN: "attribute-distance-aggregator",
    MDS_PLUGIN: "attribute-distance-mds",
    VECTOR_CONCAT_PLUGIN: "vector-concat",
}

# Per-attribute pipeline options shown in the routing step.
PIPELINE_OPTIONS = {
    NONE_PLUGIN: "None",
    WU_PALMER_PLUGIN: "Wu-Palmer",
    ONE_HOT_PLUGIN: "One-Hot",
    MAPPING_PLUGIN: "Mapping",
}

PIPELINE_FIELD_PREFIX = "pipeline_"


# This Enum class is copied from the mapping distances plugin.
# Check the mapping distances plugin for updates
class DistanceMetricEnum(Enum):
    euclidean = "Euclidean"
    manhatten = "Manhatten"
    chebyshev = "Chebyshev"
    cosine = "Cosine"


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


# This Enum class is copied from the mds plugin.
# Check the mds plugin for updates
class MetricEnum(Enum):
    metric_mds = "Metric MDS"
    nonmetric_mds = "Nonmetric MDS"


# This Enum class is copied from the mds plugin.
# Check the mds plugin for updates
class MissingDataHandling(Enum):
    mean = "Replace with mean distance"
    max = "Replace with maximum distance"


class InputParameters:
    def __init__(
        self,
        entities_url: str,
        entities_metadata_url: str,
        taxonomies_zip_url: str,
        root_is_part_of_hierarchy: bool,
        distance_metric: DistanceMetricEnum,
        transformer: TransformersEnum,
        dimensions: int,
        metric: MetricEnum,
        n_init: int,
        max_iter: int,
        missing_data_handling: MissingDataHandling,
        concat_output: bool,
        output_format: ma.fields.String,
        include_intermediate_results_in_output: bool,
    ):
        self.entities_url = entities_url
        self.entities_metadata_url = entities_metadata_url
        self.taxonomies_zip_url = taxonomies_zip_url
        self.root_is_part_of_hierarchy = root_is_part_of_hierarchy
        self.distance_metric = distance_metric
        self.transformer = transformer
        self.dimensions = dimensions
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.missing_data_handling = missing_data_handling
        self.concat_output = concat_output
        self.output_format = output_format
        self.include_intermediate_results_in_output = (
            include_intermediate_results_in_output
        )


class InputParametersSchema(FrontendFormBaseSchema):
    # Base Inputs
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/list",
        data_content_types=["text/csv", "application/json"],
        metadata={
            "label": "Entities URL",
            "description": "**[General Input]** URL to the entity list (e.g., subparts.csv).",
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
            "description": "**[General Input]** URL to a file with the attribute metadata for the entities.",
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
            "description": "**[General Input]** URL to zip file with taxonomies.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "pre",
        },
    )

    # Pipeline Specific Inputs

    root_is_part_of_hierarchy = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Consider root node as part of the hierarchy",
            "description": "**[Wu-Palmer Setting]** If the root node is part of the hierarchy, then items that are direct descendants of the "
            "root node are considered similar to a certain degree. Otherwise they will be considered as not similar. "
            "e.g. when the root node of a color taxonomy also represents a color, it should be considered as part of "
            "the hierarchy",
            "input_type": "checkbox",
        },
    )

    distance_metric = EnumField(
        DistanceMetricEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Distance Metric",
            "description": textwrap.dedent(
                r"""
                **[Mapping Setting]** Metric to calculate the distances of the taxanomy mapping:  
                **Euclidean Distance:** Length of vector (L2 norm) between two vectors: $||a-b|| = \sqrt{\sum\limits_{i} (a_i - b_i)^2}$  
                **Manhattan Distance:** Sum of distances on each vector axis: $\sum\limits_{i} |a_i - b_i|$  
                **Chebyshev Distance:** Maximum distance on one axis: $\max(|a_1 - b_1|, \dots, |a_n - b_n|)$  
                **Cosine Distance:** 1 - angle between two vectors (value in [0, 2]): $1 - \cos(\theta) = 1 - \frac{a \cdot b}{||a||\cdot||b||}$
            """
            ).strip(),
            "input_type": "select",
        },
    )

    transformer = EnumField(
        TransformersEnum,
        required=True,
        metadata={
            "label": "Transformer",
            "description": "**[Transformer Setting]** Transformer that shall be used to transform the similarities to distances.",
            "input_type": "select",
        },
    )

    dimensions = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=ma.validate.Range(min=1),
        metadata={
            "label": "Dimensions",
            "description": "**[MDS Setting]** Number of dimensions each output embedding will have.",
            "input_type": "text",
        },
    )

    metric = EnumField(
        MetricEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Metric",
            "description": (
                "**[MDS Setting]** Type of MDS that will be used. For nonmetric MDS, distances of "
                "exactly 0 are replaced with a small positive value below the "
                "smallest positive distance because scikit-learn treats them "
                "as missing values."
            ),
            "input_type": "select",
        },
    )

    n_init = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=ma.validate.Range(min=1),
        metadata={
            "label": "SMACOF executions",
            "description": "**[MDS Setting]** Number of times SMACOF will be executed with different initial values.",
            "input_type": "text",
        },
    )

    max_iter = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=ma.validate.Range(min=1),
        metadata={
            "label": "SMACOF max iterations",
            "description": "**[MDS Setting]** Maximum number of SMACOF iterations.",
            "input_type": "text",
        },
    )

    missing_data_handling = EnumField(
        MissingDataHandling,
        required=True,
        allow_none=False,
        metadata={
            "label": "Missing distances",
            "description": (
                "**[MDS Setting]** How missing (null) distances are replaced before MDS. "
                "The replacement is computed from the known distances of the same attribute."
            ),
            "input_type": "select",
        },
    )

    concat_output = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Concat output",
            "description": "**[Vector concat Setting]** If checked, the MDS output of all pipelines will be concatenated to one vector.",
            "input_type": "checkbox",
        },
    )

    output_format = ma.fields.String(
        missing="csv",
        validate=ma.validate.OneOf(("csv", "json", "lines")),
        metadata={
            "label": " Output Format",
            "description": "**[Vector concat Setting]** Format of the output data.",
            "input_type": "select",
            "options": {
                "csv": "CSV",
                "json": "JSON",
                "lines": "JSON Lines",
            },
        },
    )

    include_intermediate_results_in_output = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Include intermediate results",
            "description": "**[General Setting]** If checked, the intermediate plugin results (e.g. Wu-Palmer) will be included in the output.",
            "input_type": "checkbox",
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
            if value and value not in PIPELINE_OPTIONS.keys():
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
