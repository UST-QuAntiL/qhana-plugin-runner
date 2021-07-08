# Copyright 2021 QHAna plugin runner contributors.
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

from typing import Optional

from flask import Flask
from marshmallow import fields


def marshmallow_field_to_input_type(field: fields.Field) -> Optional[str]:
    if field.metadata.get("input_type"):
        return field.metadata.get("input_type")
    if isinstance(field, fields.Email):
        return "email"
    if isinstance(field, fields.Url):
        return "url"
    if isinstance(field, fields.Boolean):
        return "checkbox"
    if (
        isinstance(field, fields.Decimal)
        or isinstance(field, fields.Float)
        or isinstance(field, fields.Integer)
    ):
        return "number"
    if isinstance(field, fields.Date):
        return "date"
    if isinstance(field, fields.Time):
        return "time"
    if isinstance(field, fields.NaiveDateTime):
        return "datetime-local"
    if isinstance(field, fields.Raw):
        return "textarea"
    if isinstance(field, fields.String):
        return "text"  # TODO differentiate better
    return None


def register_helpers(app: Flask):
    app.jinja_env.globals["get_input_type"] = marshmallow_field_to_input_type
