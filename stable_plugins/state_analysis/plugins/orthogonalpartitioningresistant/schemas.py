import marshmallow as ma
from common.plugin_utils.marshmallow_util import (
    QasmInputList,
    SetOfComplexVectorsField,
    ToleranceField,
)
from common.plugin_utils.shemas_util import qasmInputList_util
from qhana_plugin_runner.api.util import FrontendFormBaseSchema


class Schema(FrontendFormBaseSchema):

    vectors = SetOfComplexVectorsField(
        required=False,
        metadata={
            "label": "Input Vectors",
            "description": (
                "A set of complex vectors represented as nested lists. "
                "Example: [[[1.0, 0.0], [1.0, 0.0]], ...]."
            ),
            "input_type": "textarea",
        },
    )

    qasmInputList = qasmInputList_util

    tolerance = ToleranceField(
        required=False,
        default_tolerance=1e-10,
        metadata={
            "label": "Inner Product Tolerance",
            "description": (
                "Defines the numerical threshold for considering two vectors as orthogonal "
                "based on their inner product. If omitted, the default value is 1e-10."
            ),
            "input_type": "text",
        },
    )

    @ma.post_load
    def validate_data(self, data, **kwargs):
        """
        Ensures that either 'vectors' or 'qasmInputList' is provided, but not both.
        Cleans up unused fields for consistency.
        """

        # Normalize tolerance values
        data["tolerance"] = data.get("tolerance") or None

        vectors = data.get("vectors")
        qasmInputList = data.get("qasmInputList")

        # Ensure only one input type is provided
        if vectors and qasmInputList:
            raise ValueError(
                "Only one input type is allowed: either 'vectors' or 'qasmInputList', not both."
            )
        if not vectors and not qasmInputList:
            raise ValueError(
                "At least one input is required: provide either 'vectors' or 'qasmInputList'."
            )

        # Assign None to unused fields
        if vectors:
            data["qasmInputList"] = None
        else:
            data["vectors"] = None

        return data
