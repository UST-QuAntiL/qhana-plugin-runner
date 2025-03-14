# Copyright 2025 QHAna plugin runner contributors.
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
# limitations under the Licens

import marshmallow as ma

from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


class DataJoinBaseParametersSchema(FrontendFormBaseSchema):
    base = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/*",
        data_content_types=["text/csv", "application/json"],
        metadata={
            "label": "Base Entities URL",
            "description": "The Entities to join other attributes to.",
        },
    )
