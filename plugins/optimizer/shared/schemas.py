from attr import dataclass
import marshmallow as ma
import numpy as np

from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema


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
class CalcInputData:
    x: np.ndarray
    y: np.ndarray
    x0: np.ndarray
    hyperparameters: dict


class CalcInputDataSchema(MaBaseSchema):
    x: NumpyArray = NumpyArray(required=True, allow_none=False)
    y: NumpyArray = NumpyArray(required=True, allow_none=False)
    x0: NumpyArray = NumpyArray(required=True, allow_none=False)
    hyperparameters = ma.fields.Dict(
        keys=ma.fields.String(), values=ma.fields.Float(), required=False, allow_none=True
    )

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CalcInputData(**data)
