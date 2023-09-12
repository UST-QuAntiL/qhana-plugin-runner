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
class GradientResponseData:
    gradient: np.ndarray


class GradientResponseSchema(MaBaseSchema):
    gradient = NumpyArray(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return GradientResponseData(**data)


@dataclass
class LossAndGradientResponseData:
    loss: float
    gradient: np.ndarray


class LossAndGradientResponseSchema(MaBaseSchema):
    loss = ma.fields.Float(required=True, allow_none=False)
    gradient = NumpyArray(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return LossAndGradientResponseData(**data)


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
class CalcLossOrGradInput:
    x0: np.ndarray
    x: np.ndarray
    y: np.ndarray
    hyperparameters: dict


class CalcLossOrGradInputSchema(MaBaseSchema):
    x: NumpyArray = NumpyArray(required=True, allow_none=False)
    y: NumpyArray = NumpyArray(required=True, allow_none=False)
    x0: NumpyArray = NumpyArray(required=True, allow_none=False)
    hyperparameters: dict = ma.fields.Dict(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CalcLossOrGradInput(**data)


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
    hyperparameters: dict


class ObjectiveFunctionInvokationCallbackSchema(MaBaseSchema):
    hyperparameters = ma.fields.Dict(required=True, allow_none=False)

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
    x: np.ndarray
    y: np.ndarray
    hyperparameters: dict
    calc_loss_endpoint_url: str
    calc_gradient_endpoint_url: Optional[str] = None
    calc_loss_and_gradient_endpoint_url: Optional[str] = None
    callback_url: Optional[str] = None


class MinimizerInputSchema(MaBaseSchema):
    x0 = NumpyArray(required=True, allow_none=False)
    x = NumpyArray(required=True, allow_none=False)
    y = NumpyArray(required=True, allow_none=False)
    hyperparameters = ma.fields.Dict(required=True, allow_none=False)
    calc_loss_endpoint_url = ma.fields.Url(required=True, allow_none=False)
    calc_gradient_endpoint_url = ma.fields.Url(required=False, allow_none=True)
    calc_loss_and_gradient_endpoint_url = ma.fields.Url(required=False, allow_none=True)
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
    hyperparameters: dict


class ObjectiveFunctionPassDataSchema(MaBaseSchema):
    x = NumpyArray(required=True, allow_none=False)
    y = NumpyArray(required=True, allow_none=False)
    hyperparameters = ma.fields.Dict(required=True, allow_none=False)

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
