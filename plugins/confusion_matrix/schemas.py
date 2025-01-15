# Copyright 2022 QHAna plugin runner contributors.
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


from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
)
import marshmallow as ma

class ConfusionMatrixInputParametersSchema(FrontendFormBaseSchema):
    clusters_url1 = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/label",
        data_content_types=["application/json"],
        metadata={
            "label": "Cluster URL",
            "description": "URL to a json file containing the first set of cluster labels.",
            "input_type": "text",
        },
    )
    clusters_url2 = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/label",
        data_content_types=["application/json"],
        metadata={
            "label": "Cluster URL",
            "description": "URL to a json file containing the second set of cluster labels.",
            "input_type": "text",
        },
    )
    optimize = ma.fields.Boolean(
        required=True,
        allow_none=False,
        metadata={
            "label": "Optimize",
            "description": "Optimize the Confusion Matrix by reordering the labels.",
        },
    )