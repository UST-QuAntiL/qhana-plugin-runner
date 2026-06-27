import marshmallow as ma
from marshmallow import post_load
from enum import Enum
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, FileUrl
from qhana_plugin_runner.api.extra_fields import EnumField


class RoutingOptions(Enum):
    wu_palmer = "Wu-Palmer"
    one_hot = "One-Hot"
    nummeric_mapping = "Numeric Mapping"


class MerginPoint(Enum):
    TODO = "TODO"


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
        attributes: str,
        routing_options: RoutingOptions,
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
        self.attributes = attributes
        self.routing_options = routing_options
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
            "relation": "post",
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
