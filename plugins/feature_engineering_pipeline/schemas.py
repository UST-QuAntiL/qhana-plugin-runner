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

import textwrap
from enum import Enum
from typing import Mapping

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
PCA_PLUGIN = "pca"
NORMALIZATION_PLUGIN = "normalization"

FEATURE_VECTOR = "feature_vector"
NUMERIC_MAPPING_PIPELINE = "numeric_mapping"

FINALIZE_PIPELINE = "finalize"

# Names of the plugins invoked by the routing pipeline. The runner serves
# plugin metadata at ``/plugins/<name>/`` and redirects a bare name to the
# newest installed version. The route handler turns these into external
# metadata urls (see routes.py), from which the process endpoint is resolved
# through ``get_plugin_endpoint``.
PIPELINE_PLUGINS = {
    WU_PALMER_PLUGIN: "wu-palmer",
    MAPPING_PLUGIN: "mapping-distances",
    ONE_HOT_PLUGIN: "one-hot-encoding",
    TRANSFORMERS_PLUGIN: "element_sim-to-element_dist-transformers",
    AGGREGATOR_PLUGIN: "attribute-distance-aggregator",
    MDS_PLUGIN: "attribute-distance-mds",
    VECTOR_CONCAT_PLUGIN: "vector-concat",
    PCA_PLUGIN: "pca",
    NORMALIZATION_PLUGIN: "numeric-value-normalization",
}

# Per-attribute pipeline options shown in the routing step.
PIPELINE_OPTIONS = {
    NONE_PLUGIN: "None",
    WU_PALMER_PLUGIN: "Wu-Palmer",
    ONE_HOT_PLUGIN: "One-Hot",
    MAPPING_PLUGIN: "Mapping",
}

# Value submitted by a checked numeric attribute checkbox in the routing step.
# The server picks the pipeline from the attribute metadata.
INCLUDE_NUMERIC = "numeric"

PIPELINE_FIELD_PREFIX = "pipeline_"

# Separates the attribute from the setting in the routing step field name
# ``pipeline_<attribute>__<setting>``.
SETTINGS_FIELD_SEPARATOR = "__"


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


# This Enum class is copied from the pca plugin.
# Check the pca plugin for updates
class SolverEnum(Enum):
    auto = "auto"
    full = "full"
    arpack = "arpack"
    randomized = "randomized"


# This Enum class is copied from the pca plugin.
# Check the pca plugin for updates
class PCATypeEnum(Enum):
    normal = "normal"
    incremental = "incremental"
    sparse = "sparse"
    kernel = "kernel"


class InputParameters:
    def __init__(
        self,
        # Basic Data
        entities_url: str,
        entities_metadata_url: str,
        taxonomies_zip_url: str,
        include_intermediate_results_in_output: bool,
        # Wu-Palmer Settings
        root_is_part_of_hierarchy: bool,
        # Mapping Settings
        distance_metric: DistanceMetricEnum,
        # Transformer Settings
        transformer: TransformersEnum,
        # MDS Settings
        mds_dimensions: int,
        metric: MetricEnum,
        n_init: int,
        max_iter: int,
        missing_data_handling: MissingDataHandling,
        # Vector Concatenation Settings
        concat_output: bool,
        output_format: ma.fields.String,
        # PCA Settings
        reduce_dimensions: bool,
        pca_type: PCATypeEnum,
        pca_dimensions: int,
        solver: SolverEnum,
        tol: float,
        iterated_power: int,
        # Normalization Settings
        normalization_input_range_min: float | None,
        normalization_input_range_max: float | None,
        normalization_output_range_start: float,
        normalization_output_range_end: float,
        normalization_use_clipping: bool,
        normalization_allow_missing_values: bool,
    ):
        self.entities_url = entities_url
        self.entities_metadata_url = entities_metadata_url
        self.taxonomies_zip_url = taxonomies_zip_url
        self.include_intermediate_results_in_output = (
            include_intermediate_results_in_output
        )
        self.root_is_part_of_hierarchy = root_is_part_of_hierarchy
        self.distance_metric = distance_metric
        self.transformer = transformer
        self.mds_dimensions = mds_dimensions
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.missing_data_handling = missing_data_handling
        self.concat_output = concat_output
        self.output_format = output_format
        self.reduce_dimensions = reduce_dimensions
        self.pca_type = pca_type
        self.pca_dimensions = pca_dimensions
        self.solver = solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.normalization_input_range_min = normalization_input_range_min
        self.normalization_input_range_max = normalization_input_range_max
        self.normalization_output_range_start = normalization_output_range_start
        self.normalization_output_range_end = normalization_output_range_end
        self.normalization_use_clipping = normalization_use_clipping
        self.normalization_allow_missing_values = normalization_allow_missing_values


class OptionalFloat(ma.fields.Float):
    """Float field that reads the empty string of a cleared number input as ``None``."""

    def deserialize(self, value, attr=None, data=None, **kwargs):
        if value == "":
            value = None
        return super().deserialize(value, attr, data, **kwargs)


class PCAParametersSchema(FrontendFormBaseSchema):
    """PCA settings of the routing step.

    The number of dimensions of the concatenated vector follows from the
    attribute selection, so these fields are rendered in the routing step and
    merged into the input parameters when that step is submitted. Every field
    has a default because the routing step omits the whole section when the
    pipeline outputs are not concatenated.
    """

    reduce_dimensions = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Reduce dimensions with principal component analysis (PCA)",
            "description": "If checked, the dimensions of the concatenated vector will be reduced.",
            "input_type": "checkbox",
        },
    )

    pca_type = EnumField(
        PCATypeEnum,
        load_default=PCATypeEnum.normal,
        allow_none=False,
        metadata={
            "label": "PCA Type",
            "description": "Type of PCA that will be executed.",
            "input_type": "select",
        },
    )

    pca_dimensions = ma.fields.Integer(
        load_default=1,
        allow_none=False,
        metadata={
            "label": "Dimensions",
            "description": "Number of dimensions k that the output will have."
            "\nFor k <= 0, normal PCA will guess k and all other PCA types will take max k.",
            "input_type": "number",
        },
    )

    solver = EnumField(
        SolverEnum,
        load_default=SolverEnum.auto,
        allow_none=False,
        metadata={
            "label": "Solver",
            "description": "Type of PCA solver that will be used.",
            "input_type": "select",
        },
    )

    tol = ma.fields.Float(
        load_default=0.0,
        allow_none=False,
        metadata={
            "label": "Error Tolerance",
            "description": "Tolerance (tol) for the stopping condition of arpack and of sparse PCA. \n"
            "If tol <= 0, then arpack will choose the optimal value automatically and for sparse PCA, it gets set to 1e-8.",
            "input_type": "number",
        },
    )

    iterated_power = ma.fields.Integer(
        load_default=0,
        allow_none=False,
        metadata={
            "label": "Iterated Power",
            "description": "This sets the iterated power parameter for the randomized solver. \n"
            "If it is set to <= 0, the iterated power will be chosen automatically.",
            "input_type": "number",
        },
    )


class PipelineSettingsSchema(FrontendFormBaseSchema):
    """Settings of the plugins a taxonomy attribute is routed through.

    The first step sets them for the whole run. The routing step renders the same
    fields once per attribute, so an attribute can override them.
    """

    root_is_part_of_hierarchy = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Consider root node as part of the hierarchy",
            "description": "If the root node is part of the hierarchy, then items that are direct descendants of the "
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
                Metric to calculate the distances of the taxanomy mapping:  
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
            "description": "Transformer that shall be used to transform the similarities to distances.",
            "input_type": "select",
        },
    )

    mds_dimensions = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=ma.validate.Range(min=1),
        metadata={
            "label": "Dimensions",
            "description": "Number of dimensions each output embedding will have.",
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
                "Type of MDS that will be used. For nonmetric MDS, distances of "
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
            "description": "Number of times SMACOF will be executed with different initial values.",
            "input_type": "text",
        },
    )

    max_iter = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=ma.validate.Range(min=1),
        metadata={
            "label": "SMACOF max iterations",
            "description": "Maximum number of SMACOF iterations.",
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
                "How missing (null) distances are replaced before MDS. "
                "The replacement is computed from the known distances of the same attribute."
            ),
            "input_type": "select",
        },
    )

    # These settings mirror the inputs of the numeric value normalization plugin.
    normalization_input_range_min = OptionalFloat(
        required=False,
        allow_none=True,
        load_default=None,
        metadata={
            "label": "Input range minimum",
            "description": "Manual minimum applied to every normalized value. If empty, the minimum of each value column is used.",
            "input_type": "number",
        },
    )

    normalization_input_range_max = OptionalFloat(
        required=False,
        allow_none=True,
        load_default=None,
        metadata={
            "label": "Input range maximum",
            "description": "Manual maximum applied to every normalized value. If empty, the maximum of each value column is used.",
            "input_type": "number",
        },
    )

    normalization_output_range_start = ma.fields.Float(
        required=False,
        allow_none=False,
        load_default=0.0,
        metadata={
            "label": "Output range start",
            "description": "Value the input range minimum is mapped to. Use a start greater than the end to invert the values.",
            "input_type": "number",
        },
    )

    normalization_output_range_end = ma.fields.Float(
        required=False,
        allow_none=False,
        load_default=1.0,
        metadata={
            "label": "Output range end",
            "description": "Value the input range maximum is mapped to. Use an end smaller than the start to invert the values.",
            "input_type": "number",
        },
    )

    normalization_use_clipping = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Use Clipping",
            "description": "Only matters with a manual input range, because a range taken from the data contains every value. "
            "If checked, a value outside the input range is set to the nearest output range bound "
            "(e.g. with the output range [0..1], a value below the input minimum becomes 0). "
            "If unchecked, such a value makes the normalization step fail with an error that names the entity, and the pipeline stops.",
            "input_type": "checkbox",
        },
    )

    normalization_allow_missing_values = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Allow missing values",
            "description": "Controls `null` values in the vectors. "
            "If checked, a `null` value stays `null` in the normalized vectors and is ignored when the input range is taken from the data. "
            "If unchecked, a `null` value makes the normalization step fail with an error that names the entity, and the pipeline stops.",
            "input_type": "checkbox",
        },
    )

    @ma.validates_schema
    def validate_normalization_ranges(self, data, **kwargs):
        errors = {}
        start = data.get("normalization_output_range_start")
        end = data.get("normalization_output_range_end")
        if start is not None and start == end:
            errors[self.fields["normalization_output_range_end"].data_key] = [
                f"The output range start and end must not be equal (both are {start})."
            ]
        minimum = data.get("normalization_input_range_min")
        maximum = data.get("normalization_input_range_max")
        if minimum is not None and maximum is not None and minimum >= maximum:
            errors[self.fields["normalization_input_range_max"].data_key] = [
                f"The input range maximum ({maximum}) must be greater than the "
                f"input range minimum ({minimum})."
            ]
        if errors:
            raise ma.ValidationError(errors)


class AttributeSettingsSchema(PipelineSettingsSchema):
    """Settings of one attribute in the routing step.

    Normalization is enabled per attribute, so the toggle is not part of the first step.
    """

    normalize = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Normalize",
            "description": "If checked, the feature-vector dimensions of this attribute are normalized "
            "before they are concatenated. A single-valued numeric attribute is normalized directly.",
            "input_type": "checkbox",
        },
    )


class InputParametersSchema(PipelineSettingsSchema, PCAParametersSchema):
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

    include_intermediate_results_in_output = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Include intermediate results",
            "description": "If checked, the intermediate plugin results (e.g. Wu-Palmer) will be included in the output.",
            "input_type": "checkbox",
        },
    )

    concat_output = ma.fields.Boolean(
        required=False,
        load_default=False,
        metadata={
            "label": "Concat output",
            "description": "If checked, the multidimensional scaling (MDS) output of all pipelines "
            "will be concatenated to one vector. The principal component analysis (PCA) "
            "settings are available at the end of the next step, where the attributes are "
            "selected.",
            "input_type": "checkbox",
        },
    )

    output_format = ma.fields.String(
        load_default="csv",
        validate=ma.validate.OneOf(("csv", "json", "lines")),
        metadata={
            "label": " Output Format",
            "description": "Format of the output data.",
            "input_type": "select",
            "options": {
                "csv": "CSV",
                "json": "JSON",
                "lines": "JSON Lines",
            },
        },
    )

    @ma.validates_schema
    def validate_pca_requires_concat(self, data, **kwargs):
        # PCA only ever runs on the concatenated vector, so reducing without
        # concatenating would silently skip the whole PCA step.
        if data.get("reduce_dimensions") and not data.get("concat_output"):
            raise ma.ValidationError(
                "Dimension reduction runs on the concatenated vector, "
                "so 'Concat output' has to be enabled as well.",
                self.fields["reduce_dimensions"].data_key,
            )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


PIPELINE_SETTINGS_GROUPS = (
    (TRANSFORMERS_PLUGIN, "Transformer Settings", ("transformer",)),
    (WU_PALMER_PLUGIN, "Wu-Palmer Settings", ("root_is_part_of_hierarchy",)),
    (MAPPING_PLUGIN, "Mapping Settings", ("distance_metric",)),
    (
        MDS_PLUGIN,
        "Multidimensional Scaling (MDS) Settings",
        ("mds_dimensions", "metric", "n_init", "max_iter", "missing_data_handling"),
    ),
    (
        NORMALIZATION_PLUGIN,
        "Normalization Settings",
        (
            "normalization_input_range_min",
            "normalization_input_range_max",
            "normalization_output_range_start",
            "normalization_output_range_end",
            "normalization_use_clipping",
            "normalization_allow_missing_values",
        ),
    ),
)

_SETTINGS_GROUP_FIELDS = {key: fields for key, _, fields in PIPELINE_SETTINGS_GROUPS}

NORMALIZE_FIELD = "normalize"
NORMALIZE_KEY = AttributeSettingsSchema().fields[NORMALIZE_FIELD].data_key


def _attribute_settings_groups(*group_keys: str) -> tuple:
    """Return the routing step sections of an attribute.

    The normalization section starts with the per-attribute toggle.
    """
    return tuple(
        (
            key,
            title,
            (NORMALIZE_FIELD, *fields) if key == NORMALIZATION_PLUGIN else fields,
        )
        for key, title, fields in PIPELINE_SETTINGS_GROUPS
        if key in group_keys
    )


TAXONOMY_SETTINGS_GROUPS = _attribute_settings_groups(*_SETTINGS_GROUP_FIELDS)

NUMERIC_SETTINGS_GROUPS = _attribute_settings_groups(
    MAPPING_PLUGIN, MDS_PLUGIN, NORMALIZATION_PLUGIN
)

SINGLE_VALUED_NUMERIC_SETTINGS_GROUPS = _attribute_settings_groups(NORMALIZATION_PLUGIN)

_RECOMMENDED_SETTINGS_GROUPS = {
    WU_PALMER_PLUGIN: (TRANSFORMERS_PLUGIN, WU_PALMER_PLUGIN),
    MAPPING_PLUGIN: (MAPPING_PLUGIN,),
}


def expanded_settings_groups(recommended: str | None) -> frozenset[str]:
    """Return the section keys that open by default for a recommendation."""
    return frozenset((MDS_PLUGIN, *_RECOMMENDED_SETTINGS_GROUPS.get(recommended, ())))


def expanded_numeric_settings_groups() -> frozenset[str]:
    """Return the section keys that open by default for a numeric attribute.

    The pipeline of a numeric attribute is fixed, so its sections open like a
    recommendation. Normalization is the main setting of a numeric attribute, so its
    section opens as well.
    """
    return expanded_settings_groups(MAPPING_PLUGIN) | {NORMALIZATION_PLUGIN}


_SETTINGS_FIELDS = PipelineSettingsSchema().fields


def _form_keys(*group_keys: str) -> tuple[str, ...]:
    return tuple(
        _SETTINGS_FIELDS[name].data_key
        for key in group_keys
        for name in _SETTINGS_GROUP_FIELDS[key]
    )


PIPELINE_SETTINGS_KEYS = {
    WU_PALMER_PLUGIN: _form_keys(WU_PALMER_PLUGIN, TRANSFORMERS_PLUGIN, MDS_PLUGIN),
    MAPPING_PLUGIN: _form_keys(MAPPING_PLUGIN, MDS_PLUGIN),
    NUMERIC_MAPPING_PIPELINE: _form_keys(MAPPING_PLUGIN, MDS_PLUGIN),
}

# Settings that only separate the pipeline groups of attributes that are normalized.
NORMALIZATION_SETTINGS_KEYS = _form_keys(NORMALIZATION_PLUGIN)

_ATTRIBUTE_SETTINGS_FIELDS = AttributeSettingsSchema().fields

SETTINGS_KEYS = frozenset(field.data_key for field in _ATTRIBUTE_SETTINGS_FIELDS.values())

BOOLEAN_SETTINGS_KEYS = tuple(
    field.data_key
    for field in _ATTRIBUTE_SETTINGS_FIELDS.values()
    if isinstance(field, ma.fields.Boolean)
)


def split_routing_fields(form: Mapping) -> tuple[dict[str, str], dict[str, dict]]:
    """Split a routing step form into pipeline selections and per-attribute settings.

    Returns the selections as ``{attribute: pipeline}`` and the raw settings as
    ``{attribute: {form key: submitted value}}``. Fields that belong to neither are
    ignored, ``RoutingStepParametersSchema`` rejects them beforehand.

    Attribute names contain the separator themselves (``hc__grundton``), so a field
    name is only read as a setting when it ends in a known setting. An attribute
    whose name ends in one of them, such as ``x__metric``, is not supported.
    """
    selections: dict = {}
    settings: dict = {}
    for key, value in form.items():
        if not key.startswith(PIPELINE_FIELD_PREFIX):
            continue
        name = key[len(PIPELINE_FIELD_PREFIX) :]
        attribute, separator, setting = name.rpartition(SETTINGS_FIELD_SEPARATOR)
        if separator and setting in SETTINGS_KEYS:
            settings.setdefault(attribute, {})[setting] = value
        else:
            selections[name] = value
    return selections, settings


def _with_unchecked_boxes(raw: Mapping) -> dict:
    # An unchecked checkbox is not submitted at all, so a boolean missing from a
    # submitted block is a deselected one.
    return {**{key: False for key in BOOLEAN_SETTINGS_KEYS}, **raw}


def load_settings(raw: Mapping) -> dict:
    """Validate one settings block and return it in form format.

    A setting that is not submitted keeps the value of the first step, so the block
    is loaded partially.

    Raises:
        ma.ValidationError: If a submitted setting is unknown or invalid.
    """
    schema = AttributeSettingsSchema(partial=True)
    return dict(schema.dump(schema.load(_with_unchecked_boxes(raw))))


def merge_settings(base: Mapping, raw: Mapping) -> dict:
    """Apply a submitted settings block to the settings of the whole run."""
    if not raw:
        return dict(base)
    return {**base, **_with_unchecked_boxes(raw)}


class RoutingStepParametersSchema(PCAParametersSchema):
    """Second step schema.

    The form renders one dropdown per taxonomy attribute and one checkbox per
    numeric attribute, both with the field name ``pipeline_<attribute>``. A taxonomy
    attribute also carries its own copy of the pipeline settings, submitted as
    ``pipeline_<attribute>__<setting>``. The attributes are only known at runtime, so
    the fields are accepted dynamically instead of being declared statically. The
    inherited PCA settings are part of this step because the number of dimensions to
    reduce only follows from the attribute selection.
    """

    @ma.validates_schema(pass_original=True)
    def validate_entries(self, data, original_data, **kwargs):
        allowed_values = [*PIPELINE_OPTIONS, INCLUDE_NUMERIC]
        declared_keys = {field.data_key for field in self.fields.values()}
        errors = {}
        for key in original_data:
            if key in declared_keys:
                continue
            if not key.startswith(PIPELINE_FIELD_PREFIX):
                errors[key] = [
                    f"Unexpected field '{key}', only "
                    f"'{PIPELINE_FIELD_PREFIX}<attribute>' and "
                    f"'{PIPELINE_FIELD_PREFIX}<attribute>"
                    f"{SETTINGS_FIELD_SEPARATOR}<setting>' are allowed."
                ]

        selections, settings = split_routing_fields(original_data)
        for attribute, value in selections.items():
            if value and value not in allowed_values:
                errors[f"{PIPELINE_FIELD_PREFIX}{attribute}"] = [
                    f"'{value}' is not one of {allowed_values}."
                ]
        for attribute, raw in settings.items():
            prefix = f"{PIPELINE_FIELD_PREFIX}{attribute}{SETTINGS_FIELD_SEPARATOR}"
            for key, messages in (
                AttributeSettingsSchema(partial=True).validate(raw).items()
            ):
                errors[prefix + key] = messages

        if errors:
            raise ma.ValidationError(errors)

        # The partial loads of the micro frontend legitimately see an empty form.
        if self.partial:
            return
        # A checked numeric attribute counts as a selection.
        if not any(value and value != NONE_PLUGIN for value in selections.values()):
            raise ma.ValidationError(
                "Select a pipeline or a numeric attribute for at least one attribute, "
                "otherwise there is nothing to compute."
            )

    @ma.post_load(pass_original=True)
    def add_dynamic_entries(self, data, original_data, **kwargs):
        # Each attribute maps to a single pipeline selection and to at most one value
        # per setting, so a flat ``items()`` is sufficient for plain dicts and request
        # MultiDicts alike.
        for key, value in original_data.items():
            if key.startswith(PIPELINE_FIELD_PREFIX):
                data[key] = value
        return data
