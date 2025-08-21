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

from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema, MaBaseSchema


@dataclass
class CallbackUrl:
    callback: str


class CallbackUrlSchema(MaBaseSchema):
    callback = ma.fields.URL(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CallbackUrl(**data)


@dataclass
class HyperparamterInputData:
    number_of_neurons: int


class HyperparamterInputSchema(FrontendFormBaseSchema):
    number_of_neurons = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of neurons",
            "description": "Number of neurons for the neural network.",
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


class GradientResponseSchema(MaBaseSchema):
    gradient = ma.fields.List(ma.fields.Number(), required=True, allow_none=False)


class CombinedResponseSchema(MaBaseSchema):
    loss = ma.fields.Number(required=True, allow_none=False)
    gradient = ma.fields.List(ma.fields.Number(), required=True, allow_none=False)
