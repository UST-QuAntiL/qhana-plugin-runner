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
import numpy as np

from qhana_plugin_runner.api.util import MaBaseSchema


class NumpyArray(ma.fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return {"data": value.tolist(), "dtype": str(value.dtype), "shape": value.shape}

    def _deserialize(self, value, attr, data, **kwargs):
        return np.array(value["data"], dtype=value["dtype"]).reshape(value["shape"])


@dataclass
class LossResponseData:
    loss: float


class LossResponseSchema(MaBaseSchema):
    loss = ma.fields.Float(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return LossResponseData(**data)


@dataclass
class CalcLossInputData:
    x: np.ndarray
    y: np.ndarray
    x0: np.ndarray


class CalcLossInputDataSchema(MaBaseSchema):
    x: NumpyArray = NumpyArray(required=True, allow_none=False)
    y: NumpyArray = NumpyArray(required=True, allow_none=False)
    x0: NumpyArray = NumpyArray(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CalcLossInputData(**data)


@dataclass
class ObjectiveFunctionCallbackData:
    calc_loss_endpoint_url: str


class ObjectiveFunctionCallbackSchema(MaBaseSchema):
    calc_loss_endpoint_url = ma.fields.Url(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return ObjectiveFunctionCallbackData(**data)


@dataclass
class MinimizerCallbackData:
    method: str
    minimize_endpoint_url: str


class MinimizerCallbackSchema(MaBaseSchema):
    method = ma.fields.String(required=False, allow_none=True)
    minimize_endpoint_url = ma.fields.Url(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return MinimizerCallbackData(**data)


@dataclass
class MinimizerInputData:
    x: np.ndarray
    y: np.ndarray
    calc_loss_endpoint_url: str


class MinimizerInputSchema(MaBaseSchema):
    x = NumpyArray(required=True, allow_none=False)
    y = NumpyArray(required=True, allow_none=False)
    calc_loss_endpoint_url = ma.fields.Url(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return MinimizerInputData(**data)


@dataclass
class MinimizerResult:
    weights: np.ndarray


class MinimizerResultSchema(MaBaseSchema):
    weights = NumpyArray(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return MinimizerResult(**data)
