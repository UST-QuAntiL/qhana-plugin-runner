from dataclasses import dataclass
from enum import Enum

import marshmallow as ma

from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema


class MinimizerEnum(Enum):
    nelder_mead = "Nelder-Mead"
    powell = "Powell"
    cg = "CG"
    bfgs = "BFGS"
    newton_cg = "Newton-CG"
    lbfgsb = "L-BFGS-B"
    tnc = "TNC"
    cobyla = "COBYLA"
    slsqp = "SLSQP"
    trust_constr = "trust-constr"


class MinimizerTaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass
class MinimizerSetupTaskInputData:
    method: MinimizerEnum


class MinimizerSetupTaskInputSchema(FrontendFormBaseSchema):
    method = EnumField(
        MinimizerEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Minimization Method",
            "description": "The method used for minimization.",
            "input_type": "select",
        },
    )

    @ma.post_load
    def make_task_input_data(self, data, **kwargs):
        return MinimizerSetupTaskInputData(**data)
