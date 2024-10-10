# Copyright 2024 QHAna plugin runner contributors.
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

import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api.util import FrontendFormBaseSchema


@dataclass(repr=False)
class InputParameters:
    username: str
    password: str

    def __repr__(self):
        return f'{type(self).__name__}(username="{self.username}", password="***")'

    __str__ = __repr__


class InputParametersSchema(FrontendFormBaseSchema):
    username = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "MUSE4Anything Username",
            "input_type": "text",
        },
    )
    password = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Passwort",
            "input_type": "password",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
