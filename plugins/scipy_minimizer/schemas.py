# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    dogleg = "dogleg"
    trust_ncg = "trust-ncg"
    trust_krylov = "trust-krylov"
    trust_exact = "trust-exact"


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