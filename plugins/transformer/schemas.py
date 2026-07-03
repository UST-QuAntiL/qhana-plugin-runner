from enum import Enum
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    FileUrl,
)
import marshmallow as ma
from marshmallow import post_load

class TransformersEnum(Enum):
    linear_inverse = "Linear Inverse"
    exponential_inverse = "Exponential Inverse"
    gaussian_inverse = "Gaussian Inverse"
    polynomial_inverse = "Polynomial Inverse"
    square_inverse = "Square Inverse"


class InputParameters:
    def __init__(
        self,
        element_similarities_url: str,
        attributes: str,
        transformer: TransformersEnum,
    ):
        self.element_similarities_url = element_similarities_url
        self.attributes = attributes
        self.transformer = transformer


class InputParametersSchema(FrontendFormBaseSchema):
    element_similarities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="custom/element-similarities",
        data_content_types="application/zip",
        metadata={
            "label": "Element similarities URL",
            "description": "URL to a zip file with the element similarities.",
            "input_type": "text",
        },
    )
    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Elements",
            "description": "Elements for which the similarity shall be transformed to distance.",
            "input_type": "textarea",
        },
    )
    transformer = EnumField(
        TransformersEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Transformer",
            "description": "Transformer that shall be used to transform the similarities to distances.",
            "input_type": "select",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)