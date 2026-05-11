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

from common.plugin_utils.marshmallow_util import QasmInputList

# This is used in every classical plugin so to make changes more easy this is used as reference

# TODO: refactor to a list of FileUrl + metadata fields so the QHAna UI
# can offer the file-picker integration instead of a raw text.

qasmInputList_util = QasmInputList(
    required=False,
    allow_none=True,
    # TODO ui is not working if this is commented in
    # data_input_type="executable/circuit",
    # data_content_types="text/x-qasm",
    metadata={
        "label": "OpenQASM Circuit",
        "description": (
            "A list of lists: "
            " The first element is a URL to a QASM file (required). "
            " The second element is an json with qubit_intervals (may be None)."
            ' Example: [["https://example.com/qasm1.qasm", {"qubit_intervals": [[1, 5],[7, 10],[14, 15]]}],["https://example.com/qasm2.qasm"]]'
        ),
        "input_type": "textarea",
    },
)
