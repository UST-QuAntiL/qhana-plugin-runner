from dataclasses import dataclass
from enum import Enum

from marshmallow import post_load

from qhana_plugin_runner.api import MaBaseSchema, EnumField
import marshmallow as ma

from qhana_plugin_runner.api.util import FrontendFormBaseSchema


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class Optimizers(Enum):
    cobyla = "COBYLA"
    # nelder_mead = "Nelder-Mead"


@dataclass
class Hyperparameters:
    optimizer: str
    callback_url: str


class HyperparametersSchema(FrontendFormBaseSchema):
    optimizer = EnumField(
        Optimizers,
        required=True,
        allow_none=False,
        metadata={
            "label": "Optimizer.",
            "description": "Which optimizer to use.",
            "input_type": "select",
        },
    )
    callback_url = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={
            "label": "Callback URL",
            "description": "Callback URL of the optimizer plugin. Will be filled automatically when using the optimizer plugin. MUST NOT BE CHANGED!",
            "input_type": "text",
        },
    )

    @post_load
    def make_object(self, data, **kwargs):
        return Hyperparameters(**data)
