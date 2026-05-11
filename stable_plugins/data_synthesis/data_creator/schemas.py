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
from typing import TYPE_CHECKING, Optional

import marshmallow as ma
from marshmallow import ValidationError, post_load, validate, validates_schema

from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
)

from .backend.datasets import DataTypeEnum

REQUIRED_FIELDS_BY_TYPE: dict[DataTypeEnum, frozenset[str]] = {
    DataTypeEnum.two_spirals: frozenset({"noise", "turns"}),
    DataTypeEnum.checkerboard: frozenset(),  # Empty set => only the shared fields are needed.
    DataTypeEnum.blobs: frozenset({"centers"}),
    DataTypeEnum.checkerboard_3d: frozenset(),
    DataTypeEnum.blobs_3d: frozenset({"centers"}),
}


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class InputParameters:
    dataset_type: DataTypeEnum
    num_train_points: int
    num_test_points: int
    turns: Optional[float] = None
    noise: Optional[float] = None
    centers: Optional[int] = None

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
        validate=validate.Range(min=1),
        metadata={
            "label": "No. Training Points",
            "description": "Determines the size of the training dataset.",
            "input_type": "number",
        },
    )
    num_test_points = ma.fields.Integer(
        required=True,
        allow_none=False,
        validate=validate.Range(min=0),
        metadata={
            "label": "No. Test Points",
            "description": "Determines the size of the test dataset.",
            "input_type": "number",
        },
    )
    noise = ma.fields.Float(
        required=False,
        allow_none=True,
        validate=validate.Range(min=0),
        metadata={
            "label": "Noise",
            "description": "Gives the dataset creation some noise",
            "input_type": "number",
        },
    )
    turns = ma.fields.Float(
        required=False,
        allow_none=True,
        validate=validate.Range(min=0, min_inclusive=False),
        metadata={
            "label": "Turns",
            "description": "Determines the turns of the spiral dataset",
            "input_type": "number",
        },
    )
    centers = ma.fields.Integer(
        required=False,
        allow_none=True,
        validate=validate.Range(min=1),
        metadata={
            "label": "No. Centers",
            "description": "Determines the number of Blobs",
            "input_type": "number",
        },
    )

    @validates_schema
    def validate_required_for_type(self, data, **kwargs):
        dataset_type = data.get("dataset_type")
        if not isinstance(dataset_type, DataTypeEnum):
            raise ValidationError(
                {"dataset_type": [f"Unknown dataset type: {dataset_type!r}."]}
            )
        required = REQUIRED_FIELDS_BY_TYPE.get(dataset_type, frozenset())
        missing = {f for f in required if data.get(f) is None}
        if missing:
            raise ValidationError(
                {f: ["Required for selected dataset type."] for f in sorted(missing)}
            )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)

    if TYPE_CHECKING:
        # type checking hint for tests in test_schema.py
        def load(
            self, data, *, many=None, partial=None, unknown=None
        ) -> InputParameters: ...
