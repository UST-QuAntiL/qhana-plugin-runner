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
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema, FileUrl


@dataclass
class CallbackUrl:
    callback: str


class CallbackUrlSchema(MaBaseSchema):
    callback = ma.fields.URL(required=True, allow_none=True)

    @ma.post_load
    def make_object(self, data, **kwargs):
        return CallbackUrl(**data)


class MinimizerEnum(Enum):
    nelder_mead = "Nelder-Mead"
    powell = "Powell"
    cg = "CG"
    bfgs = "BFGS"
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
        required=False,
        allow_none=True,
        metadata={
            "label": "Minimization Method",
            "description": "The method used for minimization.",
            "input_type": "select",
        },
    )

    @ma.post_load
    def make_task_input_data(self, data, **kwargs):
        return MinimizerSetupTaskInputData(**data)


class MinimizeSchema(FrontendFormBaseSchema):
    objective_function = ma.fields.URL(
        required=False,
        allow_none=False,
        metadata={
            "label": "Objective Function Task Result URL",
            "description": "The URL of an objective function task result in the evaluate step.",
            "input_type": "url",
        },
    )
    initial_weights = FileUrl(
        data_input_type="entity/vector",
        data_content_types=["text/csv", "application/json"],
        required=False,
        allow_none=True,
        metadata={
            "label": "Initial Weights",
            "description": "Preset weights to warm-start the minimization with.",
        },
    )
