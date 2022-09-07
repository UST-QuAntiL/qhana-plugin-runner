# Copyright 2022 QHAna plugin runner contributors.
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
from typing import Optional

import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api import MaBaseSchema, EnumField
from qhana_plugin_runner.api.plugin_schemas import (
    CallbackURL,
    CallbackURLSchema,
    OptimizationInput,
    OptimizationInputSchema,
)
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

    @post_load
    def make_object(self, data, **kwargs):
        return Hyperparameters(**data)


@dataclass
class InternalData:
    hyperparameters: Hyperparameters
    callback_url: CallbackURL
    optimization_input: Optional[OptimizationInput] = None


class InternalDataSchema(MaBaseSchema):
    hyperparameters = ma.fields.Nested(
        HyperparametersSchema, required=True, allow_none=False
    )
    callback_url = ma.fields.Nested(CallbackURLSchema, required=True, allow_none=False)
    optimization_input = ma.fields.Nested(
        OptimizationInputSchema(), required=False, allow_none=True
    )

    @post_load
    def make_object(self, data, **kwargs):
        return InternalData(**data)
