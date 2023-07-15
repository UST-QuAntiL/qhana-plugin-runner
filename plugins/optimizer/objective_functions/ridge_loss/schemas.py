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

import marshmallow as ma

from qhana_plugin_runner.api.util import FrontendFormBaseSchema


@dataclass
class HyperparamterInputData:
    alpha: float


class HyperparamterInputSchema(FrontendFormBaseSchema):
    alpha = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Aplha",
            "description": "Alpha variable for Ridge Loss function.",
            "input_type": "textarea",
        },
    )

    @ma.post_load
    def make_object(self, data, **kwargs):
        return HyperparamterInputData(**data)
