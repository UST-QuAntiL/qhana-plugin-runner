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


from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


class ClusterScatterInputParametersSchema(FrontendFormBaseSchema):
    entity_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=["application/json", "application/csv"],
        metadata={
            "label": "Entity Point URL",
            "description": "URL to a json file containing the points.",
            "input_type": "text",
            "related_to": "clusters_url",
            "relation": "pre",
        },
    )
    clusters_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/label",
        data_content_types=["application/json", "application/csv"],
        metadata={
            "label": "Cluster URL",
            "description": "URL to a json file containing the cluster labels.",
            "input_type": "text",
            "related_to": "entity_url",
            "relation": "post",
        },
    )
    entity_data_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/*",
        data_content_types=["application/json", "application/csv"],
        metadata={
            "label": "Original Entity Data URL",
            "description": (
                "Optional URL to original entity data (same IDs). Additional attributes "
                "are shown in hover and entity links are exposed when available."
            ),
            "input_type": "text",
            "related_to": "entity_url",
            "relation": "pre",
        },
    )
