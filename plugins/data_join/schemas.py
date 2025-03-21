# Copyright 2025 QHAna plugin runner contributors.
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
# limitations under the Licens

import marshmallow as ma
from werkzeug.datastructures import MultiDict
from webargs.multidictproxy import MultiDictProxy

from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


class DataJoinBaseParametersSchema(FrontendFormBaseSchema):
    base = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/*",
        data_content_types=["text/csv", "application/json"],
        metadata={
            "label": "Base Entities URL",
            "description": "The Entities to join other attributes to.",
        },
    )


class DataJoinJoinParametersSchema(FrontendFormBaseSchema):
    join = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/*",
        data_content_types=["text/csv", "application/json"],
        metadata={
            "label": "Select Entity URL to Join to Base",
            "description": "The Entities to join to the join base selected in the first step.",
        },
    )
    attribute = ma.fields.String(
        allow_none=False,
        required=True,
        metadata={
            "label": "Join on join.ID == ___",
            "description": "The ID attribute of the data to join will be matched against the selected attribute in the join base.",
            "input_type": "select",
        },
    )


class DataJoinFinishJoinParametersSchema(FrontendFormBaseSchema):
    pass


class DataJoinAttrSelectParametersSchema(FrontendFormBaseSchema):
    base = ma.fields.List(ma.fields.String(), default=tuple())

    @ma.validates_schema(pass_original=True)
    def validate_numbers(self, data, original_data, **kwargs):
        errors = {}
        for key in original_data:
            if key == "base" or key.startswith("join_"):
                continue
            errors[key] = [
                f"Unexpected field '{key}', only 'base' and 'join_[0-9]+' are allowed."
            ]
        if errors:
            raise ma.ValidationError(errors)

    @ma.post_load(pass_original=True)
    def add_baz_to_bar(self, data, original_data, **kwargs):
        if isinstance(original_data, (MultiDict, MultiDictProxy)):
            items = original_data.lists()
        else:
            items = original_data.items()
        for key, value in items:
            if key == "base":
                continue
            if key.startswith("join_"):
                if isinstance(value, str):
                    value = [value]
                data[key] = value
        return data
