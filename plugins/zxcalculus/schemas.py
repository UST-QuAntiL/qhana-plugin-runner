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

from marshmallow import post_load
import marshmallow as ma
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema, FileUrl
from dataclasses import dataclass


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class InputParameters:
    qubits: int
    depth: int
    simplify: bool = False

    def __str__(self):
        return str(self.__dict__)


class InputParametersSchema(FrontendFormBaseSchema):
    qubits = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "No. Qubits",
            "description": "Determines the number of qubits to generate.",
            "input_type": "number",
        },
    )
    depth = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Depth",
            "description": "Determines the depth of the circuits.",
            "input_type": "number",
        },
    )
    simplify = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Simplify",
            "description": "Simplify the generated circuit.",
            "input_type": "checkbox",
        }
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
