from dataclasses import dataclass

from marshmallow import post_load

from qhana_plugin_runner.api import MaBaseSchema
from qhana_plugin_runner.api.util import FrontendFormBaseSchema

import marshmallow as ma


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass
class Hyperparameters:
    number_of_input_values: int
    number_of_neurons: int
    callback_url: str


class HyperparametersSchema(FrontendFormBaseSchema):
    number_of_input_values = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of input values.",
            "description": "Number of input values for the neural network.",
            "input_type": "text",
        },
    )
    number_of_neurons = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of neurons",
            "description": "Number of neurons for the neural network.",
            "input_type": "text",
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
