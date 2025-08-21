# Copyright 2023 QHAna plugin runner contributors.
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

import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
)

from .backend.datasets import DataTypeEnum


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class InputParameters:
    dataset_type: DataTypeEnum
    num_train_points: int
    num_test_points: int
    turns: float = None
    noise: float = None
    centers: int = None

    def __str__(self):
        return str(self.__dict__.copy())


class InputParametersSchema(FrontendFormBaseSchema):
    dataset_type = EnumField(
        DataTypeEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Dataset Type",
            "description": """Currently available dataset types are:
- Two Spiral: Creates two spirals, spiraling out from the same point. Both spirals have a different label.
- Checkerboard: Creates a 2x2 checkerboard pattern.
- Blobs: Creates isotropic Gaussian blobs for clustering.
- 3D Checkerboard: Creates a 3D 2x2x2 checkerboard pattern.
- 3D Blobs: Creates 3D isotropic Gaussian blobs for clustering.""",
            "input_type": "select",
        },
    )
    num_train_points = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "No. Training Points",
            "description": "Determines the size of the training dataset.",
            "input_type": "number",
        },
    )
    num_test_points = ma.fields.Integer(
        required=False,
        allow_none=True,
        metadata={
            "label": "No. Test Points",
            "description": "Determines the size of the test dataset.",
            "input_type": "number",
        },
    )
    noise = ma.fields.Float(
        required=False,
        allow_none=True,
        metadata={
            "label": "Noise",
            "description": "Gives the dataset creation some noise",
            "input_type": "text",
        },
    )
    turns = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Turns",
            "description": "Determines the turns of the spiral dataset",
            "input_type": "text",
        },
    )
    centers = ma.fields.Integer(
        required=False,
        allow_none=False,
        metadata={
            "label": "No. Centers",
            "description": "Determines the number of Blobs",
            "input_type": "number",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
