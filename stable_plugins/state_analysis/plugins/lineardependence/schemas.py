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
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


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

    vectorsUrl = FileUrl(
        required=False,
        allow_none=True,
        load_default=None,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Input Vectors URL",
            "description": (
                "Optional URL to a JSON file containing the same set of complex "
                "vectors. If provided, this takes precedence over the inline "
                '"Input Vectors" textarea above and lets this plugin be chained '
                "from upstream plugins that emit entity/vector data."
            ),
            "input_type": "text",
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

        vectors_url = data.get("vectorsUrl")
        provided = [
            field
            for field in ("vectors", "vectorsUrl", "qasmInputList")
            if data.get(field)
        ]

        if len(provided) > 1:
            raise ValueError(
                "Only one input type is allowed: provide exactly one of "
                "'vectors', 'vectorsUrl', or 'qasmInputList'."
            )
        if not provided:
            raise ValueError(
                "At least one input is required: provide 'vectors', "
                "'vectorsUrl', or 'qasmInputList'."
            )

        # Assign None to unused fields so downstream code sees a stable shape
        if vectors_url:
            data["vectors"] = None
        elif data.get("vectors"):
            data["circuit"] = None
        else:
            data["vectors"] = None

        return data
