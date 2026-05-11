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
from common.plugin_utils.marshmallow_util import QasmInputList
from common.plugin_utils.schemas_util import qasmInputList_util
from marshmallow.validate import Range

from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema, PluginUrl


class Schema(FrontendFormBaseSchema):
    qasmInputList = qasmInputList_util

    executor = PluginUrl(
        required=True,
        plugin_tags=["circuit-executor", "qasm-2"],
        metadata={
            "label": "Select Circuit Executor Plugin",
        },
    )

    executionOptions = FileUrl(
        required=False,
        allow_none=True,
        load_missing=None,
        data_input_type="provenance/execution-options",
        data_content_types=["text/csv", "application/json", "application/X-lines+json"],
        metadata={
            "label": "Execution Options (optional)",
            "description": "URL to a file containing execution options. (optional)",
            "input_type": "text",
        },
    )
    shots = ma.fields.Integer(
        required=False,
        allow_none=True,
        load_default=None,
        validate=Range(min=1, min_inclusive=True),
        metadata={
            "label": "Shots",
            "description": "The number of shots to simulate. If execution options are specified they will override this setting!",
            "input_type": "number",
        },
    )

    @ma.post_load
    def validate_data(self, data, **kwargs):
        """
        Ensures that 'qasmInputList' is provided.
        """

        qasmInputList = data.get("qasmInputList")
        if not qasmInputList:
            raise ValueError("The field 'qasmInputList' is required and cannot be empty.")

        shots = data.get("shots")
        if not shots:
            raise ValueError("The field 'shots' is required and cannot be empty.")
        return data
