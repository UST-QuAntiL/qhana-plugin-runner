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
from typing import Dict, Optional, Union

import marshmallow as ma
from marshmallow import post_load, pre_load
from werkzeug.datastructures import ImmutableMultiDict

from qhana_plugin_runner.api.util import FrontendFormBaseSchema


@dataclass(repr=False)
class InputParameters:
    username: str
    password: str
    muse_url: Optional[str] = None

    def __repr__(self):
        url = "" if self.muse_url is None else f', muse_url="{self.muse_url}"'
        return f'{type(self).__name__}(username="{self.username}", password="***"{url})'

    __str__ = __repr__


class InputParametersSchema(FrontendFormBaseSchema):
    username = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "MUSE4Music Username",
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
    muse_url = ma.fields.Url(
        required=False,
        allow_none=True,
        metadata={
            "label": "MUSE4Music URL",
            "input_type": "string",
        },
    )

    @pre_load
    def prepare_for_validation(
        self, data: Union[Dict, ImmutableMultiDict], **kwargs
    ) -> dict:
        if isinstance(data, ImmutableMultiDict):
            data = data.to_dict()
        else:
            data = dict(data)
        if not data.get("museUrl"):
            data.pop("museUrl", None)
        return data

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
