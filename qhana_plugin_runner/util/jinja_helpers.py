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

from typing import List, Optional, Sequence, Union

from flask import Flask
from marshmallow import fields
from marshmallow.validate import Length, Range


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


def marshmallow_field_to_step_attr(field: fields.Field) -> Optional[str]:
    if field.metadata.get("step") is not None:
        return field.metadata.get("step")
    if isinstance(field, fields.Integer):
        return "1"
    if isinstance(field, fields.Decimal) or isinstance(field, fields.Float):
        return "any"
    return None


def marshmallow_validators_to_field_attrs(field: fields.Field) -> str:
    attrs: List[str] = []
    step = marshmallow_field_to_step_attr(field)
    if step:
        attrs.append(f"step={step}")

    for validator in field.validators:
        if isinstance(validator, Range):
            if validator.min is not None:
                attrs.append(f"min={validator.min}")
            if validator.max is not None:
                attrs.append(f"max={validator.max}")
        if isinstance(validator, Length):
            if validator.min is not None:
                attrs.append(f"minlength={validator.min}")
            if validator.max is not None:
                attrs.append(f"maxlength={validator.max}")

    if attrs:
        return " ".join(attrs)

    return ""


def space_delimited_list(
    items: Union[Sequence[Union[str, float, int, bool, None]], str, float, int, bool]
) -> Optional[str]:
    if not isinstance(items, (list, tuple)):
        if items is None:
            return None
        return str(items)
    return " ".join(str(i) for i in items)


def register_helpers(app: Flask):
    app.jinja_env.globals["get_input_type"] = marshmallow_field_to_input_type
    app.jinja_env.globals["get_input_attr_step"] = marshmallow_field_to_step_attr
    app.jinja_env.globals["get_validation_attrs"] = marshmallow_validators_to_field_attrs
    app.jinja_env.globals["space_delimited_list"] = space_delimited_list
