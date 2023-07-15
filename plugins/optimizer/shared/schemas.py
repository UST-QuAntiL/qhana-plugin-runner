from dataclasses import dataclass
from typing import Optional

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
class TaskStatusChanged:
    url: str
    status: str


class TaskStatusChangedSchema(MaBaseSchema):
    url = ma.fields.Url(required=True, allow_none=True)
    status = ma.fields.String(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return TaskStatusChanged(**data)


@dataclass
class CalcLossInput:
    x0: np.ndarray
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None


class CalcLossInputSchema(MaBaseSchema):
    x: NumpyArray = NumpyArray(required=False, allow_none=True)
    y: NumpyArray = NumpyArray(required=False, allow_none=True)
    x0: NumpyArray = NumpyArray(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CalcLossInput(**data)


@dataclass
class ObjectiveFunctionCallbackData:
    calc_loss_endpoint_url: str


class ObjectiveFunctionCallbackSchema(MaBaseSchema):
    calc_loss_endpoint_url = ma.fields.Url(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return ObjectiveFunctionCallbackData(**data)


@dataclass
class ObjectiveFunctionInvokationCallbackData:
    db_id: int


class ObjectiveFunctionInvokationCallbackSchema(MaBaseSchema):
    db_id = ma.fields.Integer(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return ObjectiveFunctionInvokationCallbackData(**data)


@dataclass
class MinimizerCallbackData:
    minimize_endpoint_url: str


class MinimizerCallbackSchema(MaBaseSchema):
    minimize_endpoint_url = ma.fields.Url(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return MinimizerCallbackData(**data)


@dataclass
class MinimizerInputData:
    x0: np.ndarray
    calc_loss_endpoint_url: str
    callback_url: Optional[str] = None


class MinimizerInputSchema(MaBaseSchema):
    x0 = NumpyArray(required=True, allow_none=False)
    calc_loss_endpoint_url = ma.fields.Url(required=True, allow_none=False)
    callback_url = ma.fields.Url(required=False, allow_none=True)

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


@dataclass
class ObjectiveFunctionPassData:
    x: np.ndarray
    y: np.ndarray


class ObjectiveFunctionPassDataSchema(MaBaseSchema):
    x = NumpyArray(required=True, allow_none=False)
    y = NumpyArray(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return ObjectiveFunctionPassData(**data)


@dataclass
class ObjectiveFunctionPassDataResponse:
    number_weights: int


class ObjectiveFunctionPassDataResponseSchema(MaBaseSchema):
    number_weights = ma.fields.Integer(
        required=True, allow_none=False, data_key="numberWeights"
    )

    @ma.post_load
    def make_object(self, data, **kwargs):
        return ObjectiveFunctionPassDataResponse(**data)


@dataclass
class SingleNumpyArray:
    array: np.ndarray


class SingleNumpyArraySchema(MaBaseSchema):
    array = NumpyArray(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return SingleNumpyArray(**data)
