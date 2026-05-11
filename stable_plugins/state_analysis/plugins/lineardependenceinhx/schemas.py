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

import marshmallow as ma
from common.plugin_utils.marshmallow_util import (
    QasmInputList,
    SetOfComplexVectorsField,
    ToleranceField,
)
from common.plugin_utils.schemas_util import qasmInputList_util
from qhana_plugin_runner.api.util import FrontendFormBaseSchema


class Schema(FrontendFormBaseSchema):

    vectors = SetOfComplexVectorsField(
        required=False,
        metadata={
            "label": "Input Vectors",
            "description": (
                "A set of complex vectors in the bipartite system `H_X ⊗ H_R`, "
                "represented as nested lists.\n\n"
                "**Example:** `[[[1.0, 0.0], [1.0, 0.0]], ...]`"
            ),
            "input_type": "textarea",
        },
    )

    qasmInputList = qasmInputList_util

    dimHX = ma.fields.Integer(
        required=True,
        metadata={
            "label": "Dimension H_X",
            "description": "Number of qubits spanning the subsystem `H_X`.",
            "input_type": "number",
        },
    )

    dimHR = ma.fields.Integer(
        required=True,
        metadata={
            "label": "Dimension H_R",
            "description": "Number of qubits spanning the subsystem `H_R`.",
            "input_type": "number",
        },
    )

    schmidtBaseTolerance = ToleranceField(
        required=False,
        default_tolerance=1e-10,
        metadata={
            "label": "Schmidt Decomposition Tolerance",
            "description": (
                "Performs the Schmidt decomposition for each input vector, identifying "
                "the Schmidt basis `|u_j,k⟩`. The decomposition is based on Singular "
                "Value Decomposition (SVD):\n\n"
                "- singular values **≥** this tolerance are kept as part of the basis.\n"
                "- singular values **<** this tolerance are dropped."
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
                "Numerical threshold for considering singular values as nonzero in "
                "the linear dependence test. The input vectors are stacked into a "
                "matrix and SVD is applied — the number of nonzero singular values "
                "is the matrix rank and determines whether the vectors are linearly "
                "dependent.\n\n"
                "- values **≥** this tolerance count as nonzero.\n"
                "- values **<** this tolerance are treated as zero."
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
