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

from enum import Enum

import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


class AggregatorsEnum(Enum):
    mean = "Mean"
    median = "Median"
    max = "Max"
    min = "Min"


class MissingDataHandling(Enum):
    ignore = "ignore"
    mean = "mean"
    max = "max"


class InputParameters:
    def __init__(
        self,
        entities_url: str,
        element_distances_url: str,
        attributes: str,
        aggregator: AggregatorsEnum,
        missing_data_handling: MissingDataHandling,
    ):
        self.entities_url = entities_url
        self.element_distances_url = element_distances_url
        self.attributes = attributes
        self.aggregator = aggregator
        self.missing_data_handling = missing_data_handling


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
    attributes = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Attributes",
            "description": "Attributes for which the element distances shall be aggregated to attribute distances.",
            "input_type": "textarea",
        },
    )
    aggregator = EnumField(
        AggregatorsEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Aggregator",
            "description": "Aggregator that shall be used to aggregate the element distances "
            "of an entity pair to a single attribute distance value.",
            "input_type": "select",
        },
    )
    missing_data_handling = EnumField(
        MissingDataHandling,
        required=True,
        metadata={
            "label": "Missing data handling",
            "description": """Defines how a missing element distance should be handled.
- ignore: null values are removed and only the not null values are used for the aggregation
- mean: null values are replaced by the mean distance of the respective attribute
- max: null values are replaced by the maximum distance of the respective attribute""",
            "input_type": "select",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
