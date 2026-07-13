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

from dataclasses import dataclass
from enum import Enum
import marshmallow as ma

from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


class DistanceMetricEnum(Enum):
    euclidean = "Euclidean"
    manhatten = "Manhatten"
    chebyshev = "Chebyshev"
    cosine = "Cosine"


@dataclass
class InputParameters:
    entities_url: str
    entities_metadata_url: str
    taxonomies_zip_url: str
    attributes: str
    distance_metric: DistanceMetricEnum


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
    entities_metadata_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/attribute-metadata",
        data_content_types=["application/json", "application/X-lines+json", "text/csv"],
        metadata={
            "label": "Entities Attribute Metadata URL",
            "description": "URL to a file with the attribute metadata for the entities.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "post",
        },
    )
    taxonomies_zip_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="graph/taxonomy",
        data_content_types="application/zip",
        metadata={
            "label": "Taxonomies URL",
            "description": "URL to zip file with taxonomies.",
            "input_type": "text",
            "related_to": "entities_url",
            "relation": "pre",
        },
    )
    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Attributes",
            "description": "List of attributes for which the similarity shall be computed. Separated by newlines.",
            "input_type": "textarea",
        },
    )
    distance_metric = EnumField(
        DistanceMetricEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Distance Metric",
            "description": (
                "Metric to calculate the distances of the taxanomy mapping.<br />"
                "**Euclidean Distance:** Length of vector (L2 norm) between two vectors: "
                "$||a-b|| = \sqrt{\sum\limits_{i} (a_i - b_i)^2}$<br />"
                "**Manhattan Distance:** Sum of distances on each vector axis: $\sum\limits_{i} |a_i - b_i|$<br />"
                "**Chebyshev Distance:** Maximum distance on one axis: $\max(|a_1 - b_1|, \dots, |a_n - b_n|)$<br />"
                "**Cosine Distance:** 1 - angle between two vectors (value in [0, 2]): "
                "$1 - \cos(\\theta) = 1 - \\frac{a \cdot b}{||a||\cdot||b||}$"
            ),
            "input_type": "select",
        },
    )

    @ma.post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
