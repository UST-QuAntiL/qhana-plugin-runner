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


@dataclass
class GradientResponseData:
    gradient: np.ndarray


@dataclass
class LossAndGradientResponseData:
    loss: float
    gradient: np.ndarray


@dataclass
class TaskStatusChanged:
    url: str
    status: str


@dataclass
class CalcLossOrGradInput:
    x0: np.ndarray
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None


@dataclass
class ObjectiveFunctionCallbackData:
    calc_loss_endpoint_url: str


@dataclass
class ObjectiveFunctionInvokationCallbackData:
    task_id: int


@dataclass
class MinimizerCallbackData:
    minimize_endpoint_url: str


@dataclass
class MinimizerInputData:
    x0: np.ndarray
    calc_loss_endpoint_url: str
    calc_gradient_endpoint_url: Optional[str] = None
    calc_loss_and_gradient_endpoint_url: Optional[str] = None
    callback_url: Optional[str] = None


@dataclass
class MinimizerResult:
    weights: np.ndarray


@dataclass
class ObjectiveFunctionPassData:
    x: np.ndarray
    y: np.ndarray


@dataclass
class ObjectiveFunctionPassDataResponse:
    number_weights: int


@dataclass
class SingleNumpyArray:
    array: np.ndarray
