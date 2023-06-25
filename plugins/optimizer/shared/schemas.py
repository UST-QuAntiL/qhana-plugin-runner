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
