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
class ObjectiveFunctionInvocationCallback:
    db_id: int


class ObjectiveFunctionInvocationCallbackSchema(MaBaseSchema):
    db_id = ma.fields.Integer(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return ObjectiveFunctionInvocationCallback(**data)


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
    x: np.ndarray
    y: np.ndarray
    calc_loss_endpoint_url: str
    callback_url: Optional[str] = None


class MinimizerInputSchema(MaBaseSchema):
    x = NumpyArray(required=True, allow_none=False)
    y = NumpyArray(required=True, allow_none=False)
    calc_loss_endpoint_url = ma.fields.Url(required=True, allow_none=False)
    callback_url = ma.fields.Url(required=False, allow_none=False)

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
    callback_url: str


class ObjectiveFunctionPassDataSchema(MaBaseSchema):
    x = NumpyArray(required=True, allow_none=False)
    y = NumpyArray(required=True, allow_none=False)
    callback_url = ma.fields.Url(required=False, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return ObjectiveFunctionPassData(**data)


@dataclass
class ObjectiveFunctionPassDataResponse:
    number_weights: int


class ObjectiveFunctionPassDataResponseSchema(MaBaseSchema):
    number_weights = ma.fields.Integer(required=True, allow_none=False)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return ObjectiveFunctionPassDataResponse(**data)


@dataclass
class MinimizationInteractionEndpointInputData:
    fun = str
    x0 = Optional[np.ndarray]
    x = np.ndarray
    y = np.ndarray


class MinimizationInteractionEndpointInputSchema(MaBaseSchema):
    fun = ma.fields.URL(
        required=True,
        allow_none=False,
        metadata={
            "description": "The URL to the objecttive function interaction endpoint."
        },
    )
    x0 = NumpyArray(
        required=False,
        allow_none=False,
        metadata={
            "description": "Initial guess. Array of real elements of size (n,), where n is the number of independent variables."
        },
    )

    x = NumpyArray(
        required=True,
        allow_none=False,
        metadata={"description": "Independent variables."},
    )

    y = NumpyArray(
        required=True,
        allow_none=False,
        metadata={"description": "Dependent variables"},
    )
