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

from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema, FileUrl


@dataclass
class HyperparamterInputData:
    c: float


class HyperparamterInputSchema(FrontendFormBaseSchema):
    c = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Regularization parameter",
            "description": "Regularization parameter for Hinge Loss function.",
            "input_type": "number",
        },
    )

    @ma.post_load
    def make_object(self, data, **kwargs):
        return HyperparamterInputData(**data)


class PassDataSchema(FrontendFormBaseSchema):
    features = FileUrl(
        data_input_type="entity/vector",
        data_content_types=["text/csv", "application/json", "application/X-lines+json"],
        required=True,
        metadata={
            "label": "Features: 2-D array",
            "description": "Each entity is 1 sample with k numeric features.",
        },
    )
    target = FileUrl(
        data_input_type="entity/vector",
        data_content_types=["text/csv", "application/json"],
        required=True,
        metadata={
            "label": "Target: 1-D array",
            "description": "Each entity is 1 sample with 1 numeric target value.",
        },
    )


class EvaluateSchema(FrontendFormBaseSchema):
    pass  # intentionally empty


class WeightsResponseSchema(MaBaseSchema):
    weights = ma.fields.Integer(required=True, allow_none=False)


class EvaluateRequestSchema(MaBaseSchema):
    weights = ma.fields.List(ma.fields.Number(), required=True, allow_none=False)


class LossResponseSchema(MaBaseSchema):
    loss = ma.fields.Number(required=True, allow_none=False)
