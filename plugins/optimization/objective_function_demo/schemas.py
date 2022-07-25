from qhana_plugin_runner.api import MaBaseSchema
from qhana_plugin_runner.api.util import FrontendFormBaseSchema

import marshmallow as ma


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


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


class CalculationInputSchema(MaBaseSchema):
    data_set = ma.fields.Url(required=True, allow_none=False)
    db_id = ma.fields.Integer(required=True, allow_none=False)
    parameters = ma.fields.List(ma.fields.Float(), required=True, allow_none=False)


class CalculationOutputSchema(MaBaseSchema):
    objective_value = ma.fields.Float(required=True, allow_none=False)
