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
                "A set of complex vectors in the bipartite system H_X ⊗ H_R, "
                "represented as nested lists. Example: [[[1.0, 0.0], [1.0, 0.0]], ...]"
            ),
            "input_type": "textarea",
        },
    )

    qasmInputList = qasmInputList_util
    # Markierung für Parameter Dim_HX und Dim_HR
    dimHX = ma.fields.Integer(
        required=True,
        metadata={
            "label": "Dimension H_X",
            "description": "Number of qubits spanning the subsystem H_X.",
            "input_type": "number",
        },
    )

    dimHR = ma.fields.Integer(
        required=True,
        metadata={
            "label": "Dimension H_R",
            "description": "Number of qubits spanning the subsystem H_R.",
            "input_type": "number",
        },
    )

    schmidtBaseTolerance = ToleranceField(
        required=False,
        default_tolerance=1e-10,
        metadata={
            "label": "Schmidt Decomposition Tolerance",
            "description": (
                "Performs the Schmidt decomposition for each state in the input vectors, identifying "
                "the Schmidt basis |u_j,k⟩. This decomposition is based on Singular Value Decomposition (SVD), "
                "where singular values greater than or equal to this tolerance are considered part of the Schmidt basis. "
                "Values below this threshold are ignored."
            ),
            "input_type": "text",
        },
    )

    linearDependenceTolerance = ToleranceField(
        required=False,
        default_tolerance=1e-10,
        metadata={
            "label": "Linear Dependence Tolerance",
            "description": (
                "Defines the numerical threshold for considering singular values as nonzero in a linear dependence test. "
                "The input vectors are arranged into a matrix, and Singular Value Decomposition (SVD) is applied. "
                "The number of nonzero singular values determines the rank of the matrix, indicating whether the vectors "
                "are linearly dependent. Values below this tolerance are treated as zero."
            ),
            "input_type": "text",
        },
    )

    @ma.post_load
    def validate_data(self, data, **kwargs):
        if data.get("schmidtBaseTolerance") in ("", None):
            data["schmidtBaseTolerance"] = None
        if data.get("linearDependenceTolerance") in ("", None):
            data["linearDependenceTolerance"] = None

        # Ensure required dimensions are provided
        if data.get("dimHX") is None or data.get("dimHR") is None:
            raise ValueError("Both 'dimHX' and 'dimHR' must be provided.")

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
