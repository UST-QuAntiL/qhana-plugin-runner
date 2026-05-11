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
    ComplexVectorField,
    QasmInputList,
    ToleranceField,
)
from common.plugin_utils.schemas_util import qasmInputList_util
from qhana_plugin_runner.api.util import FrontendFormBaseSchema


class Schema(FrontendFormBaseSchema):
    """
    Validates either:
      - A single complex vector (`vector`) along with dimensions `dimHX` and `dimHR`, or
      - A circuit descriptor (`qasmInputList`) from which exactly one vector is decoded, along with `dimHX` and `dimHR`.
    """

    vector = ComplexVectorField(
        required=False,
        metadata={
            "label": "Input Vector",
            "description": (
                "A single quantum state in the bipartite system H_X ⊗ H_R, "
                "represented as a nested list. Example: [[1.0, 0.0], [1.0, 0.0], ...]."
            ),
            "input_type": "textarea",
        },
    )

    qasmInputList = qasmInputList_util

    dimHX = ma.fields.Integer(
        required=True,
        metadata={
            "label": "Dimension H_X",
            "description": "The dimension of the subsystem H_X.",
            "input_type": "number",
        },
    )

    dimHR = ma.fields.Integer(
        required=True,
        metadata={
            "label": "Dimension H_R",
            "description": "The dimension of the subsystem H_R.",
            "input_type": "number",
        },
    )

    tolerance = ToleranceField(
        required=False,
        default_tolerance=1e-10,
        metadata={
            "label": "Schmidt Decomposition Tolerance",
            "description": (
                "Defines the threshold for singular values in Schmidt decomposition. "
                "Based on Singular Value Decomposition (SVD), values greater than or equal to "
                "this tolerance are considered part of the Schmidt basis, while smaller values are ignored. "
                "The number of such values determines the Schmidt rank."
            ),
            "input_type": "text",
        },
    )

    @ma.post_load
    def validate_data(self, data, **kwargs):
        """
        Ensures that either 'vector' or 'qasmInputList' is provided, but not both.
        Checks that 'dimHX' and 'dimHR' are always present.
        """
        if data.get("tolerance") in ("", None):
            data["tolerance"] = None

        vector = data.get("vector")
        qasmInputList = data.get("qasmInputList")

        # Ensure dimensions are provided
        if data.get("dimHX") is None or data.get("dimHR") is None:
            raise ValueError("Both 'dimHX' and 'dimHR' must be provided.")

        # Ensure only one input type is provided
        if vector and qasmInputList:
            raise ValueError(
                "Only one input type is allowed: either 'vector' or 'qasmInputList', not both."
            )
        if not vector and not qasmInputList:
            raise ValueError(
                "At least one input is required: provide either 'vector' or 'qasmInputList'."
            )

        return data
