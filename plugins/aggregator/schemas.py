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

from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


class InputParametersSchema(FrontendFormBaseSchema):
    entities_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/list",
        data_content_types=["application/json", "application/X-lines+json", "text/csv"],
        metadata={
            "label": "Entities URL",
            "description": "URL to a file with entities.",
            "input_type": "text",
        },
    )
    element_distances_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="relation/element-distances",
        data_content_types="application/zip",
        metadata={
            "label": "Element distances URL",
            "description": "URL to a zip file with the element distances for the entities.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "post",
        },
    )
