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

from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


class MetricEnum(Enum):
    metric_mds = "Metric MDS"
    nonmetric_mds = "Nonmetric MDS"


class MissingDataHandling(Enum):
    mean = "Replace with mean distance"
    max = "Replace with maximum distance"


@dataclass
class InputParameters:
    attribute_distances_url: str
    dimensions: int
    metric: MetricEnum
    n_init: int
    max_iter: int
    missing_data_handling: MissingDataHandling


class InputParametersSchema(FrontendFormBaseSchema):
    attribute_distances_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="relation/attribute-distances",
        data_content_types="application/zip",
        metadata={
            "label": "Attribute distances URL",
            "description": "URL to a zip file with the attribute distances.",
            "input_type": "text",
        },
    )
    dimensions = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=ma.validate.Range(min=1),
        metadata={
            "label": "Dimensions",
            "description": "Number of dimensions each output embedding will have.",
            "input_type": "text",
        },
    )
    metric = EnumField(
        MetricEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Metric",
            "description": "Type of MDS that will be used.",
            "input_type": "select",
        },
    )
    n_init = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=ma.validate.Range(min=1),
        metadata={
            "label": "SMACOF executions",
            "description": "Number of times SMACOF will be executed with different initial values.",
            "input_type": "text",
        },
    )
    max_iter = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=ma.validate.Range(min=1),
        metadata={
            "label": "SMACOF max iterations",
            "description": "Maximum number of SMACOF iterations.",
            "input_type": "text",
        },
    )
    missing_data_handling = EnumField(
        MissingDataHandling,
        required=True,
        allow_none=False,
        metadata={
            "label": "Missing distances",
            "description": (
                "How missing (null) distances are replaced before MDS. "
                "The replacement is computed from the known distances of the same attribute."
            ),
            "input_type": "select",
        },
    )

    @ma.post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
