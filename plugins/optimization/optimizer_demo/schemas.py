from enum import Enum

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


class OptimizationInputSchema(MaBaseSchema):
    dataset = ma.fields.Url(required=True, allow_none=False)
    optimizer_db_id = ma.fields.Integer(required=True, allow_none=False)
    number_of_parameters = ma.fields.Integer(required=True, allow_none=False)
    obj_func_db_id = ma.fields.Integer(required=True, allow_none=False)
    obj_func_calc_url = ma.fields.Url(required=True, allow_none=False)


class OptimizationOutputSchema(MaBaseSchema):
    last_objective_value = ma.fields.Float(required=True, allow_none=False)
    optimized_parameters = ma.fields.List(
        ma.fields.Float(), required=True, allow_none=False
    )
