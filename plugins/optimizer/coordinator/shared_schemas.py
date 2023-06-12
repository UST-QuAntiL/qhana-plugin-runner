from dataclasses import dataclass
import marshmallow as ma
import numpy as np

from qhana_plugin_runner.api.util import MaBaseSchema


class CallbackURLSchema(MaBaseSchema):
    callback_url = ma.fields.URL(required=True, allow_none=False)


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
    hyperparameters: dict


class CalcLossInputDataSchema(MaBaseSchema):
    x: NumpyArray = NumpyArray(required=True, allow_none=False)
    y: NumpyArray = NumpyArray(required=True, allow_none=False)
    x0: NumpyArray = NumpyArray(required=True, allow_none=False)
    hyperparameters = ma.fields.Dict(
        keys=ma.fields.String(), values=ma.fields.Float(), required=False, allow_none=True
    )

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CalcLossInputData(**data)


@dataclass
class ObjectiveFunctionCallbackData:
    hyperparameters: dict
    calc_loss_endpoint_url: str


class ObjectiveFunctionCallbackSchema(MaBaseSchema):
    hyperparameters = ma.fields.Dict(
        keys=ma.fields.String(), values=ma.fields.Float(), required=False, allow_none=True
    )
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
    hyperparameters: dict
    calc_loss_endpoint_url: str


class MinimizerInputSchema(MaBaseSchema):
    x = NumpyArray(required=True, allow_none=False)
    y = NumpyArray(required=True, allow_none=False)
    hyperparameters = ma.fields.Dict(
        keys=ma.fields.String(), values=ma.fields.Float(), required=True, allow_none=False
    )
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
