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
                "Example: [[[1.0, 0.0], [1.0, 0.0]], ...]"
            ),
            "input_type": "textarea",
        },
    )

    qasmInputList = qasmInputList_util

    tolerance = ToleranceField(
        required=False,
        default_tolerance=1e-10,
        metadata={
            "label": "Tolerance",
            "description": (
                "Defines the numerical threshold for considering singular values as nonzero. "
                "The input vectors are arranged into a matrix, and Singular Value Decomposition (SVD) "
                "is performed. The number of nonzero singular values determines the matrix rank, "
                "indicating whether the vectors are linearly dependent. "
                "Values below this tolerance are treated as zero."
            ),
            "input_type": "text",
        },
    )

    @ma.post_load
    def fix_data(self, data, **kwargs):
        if data.get("tolerance") in ("", None):
            data["tolerance"] = None

        # Ensure only one input type is provided
        if data.get("vectors") and data.get("qasmInputList"):
            raise ValueError(
                "Only one input type is allowed: either 'vectors' or 'qasmInputList', not both."
            )
        if not data.get("vectors") and not data.get("qasmInputList"):
            raise ValueError(
                "At least one input is required: provide either 'vectors' or 'qasmInputList'."
            )

        # Assign None to unused fields
        if data.get("vectors"):
            data["circuit"] = None
        else:
            data["vectors"] = None

        return data
