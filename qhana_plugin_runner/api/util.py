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

# originally from <https://github.com/buehlefs/flask-template/>

"""Module containing utilities for flask smorest APIs."""
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import marshmallow as ma
from flask_smorest import Blueprint
from marshmallow import types

from .jwt import JWTMixin


class SecurityBlueprint(Blueprint, JWTMixin):
    """Blueprint that is aware of jwt tokens and how to document them.

    Use this Blueprint if you want to document security requirements for your api.
    """

    def __init__(self, *args: Any, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_doc_cbks.append(self._prepare_security_doc)


def camelcase(s: str) -> str:
    """Turn a string from python snake_case into camelCase."""
    parts = iter(s.split("_"))
    return next(parts) + "".join(i.title() for i in parts)


class MaBaseSchema(ma.Schema):
    """Base schema that automatically changes python snake case to camelCase in json."""

    # Uncomment to get ordered output
    # class Meta:
    #    ordered: bool = True

    def on_bind_field(self, field_name: str, field_obj: ma.fields.Field):
        field_obj.data_key = camelcase(field_obj.data_key or field_name)


class FrontendFormBaseSchema(MaBaseSchema):
    """Base schema for micro frontend forms that may be partially filled or contain errors.

    Schemas inherited from this class retain their field order.

    Set ``validate_errors_as_result=True`` to get the result of the ``validate()``
    function in the endpoint instead of the parsed data.
    """

    class Meta:
        ordered = True

    def __init__(
        self,
        *,
        only: Optional[types.StrSequenceOrSet] = None,
        exclude: types.StrSequenceOrSet = (),
        many: bool = False,
        context: Optional[Dict] = None,
        load_only: types.StrSequenceOrSet = (),
        dump_only: types.StrSequenceOrSet = (),
        partial: Union[bool, types.StrSequenceOrSet] = False,
        unknown: Optional[str] = None,
        validate_errors_as_result: bool = False,
    ):
        super().__init__(
            only=only,
            exclude=exclude,
            many=many,
            context=context,
            load_only=load_only,
            dump_only=dump_only,
            partial=partial,
            unknown=unknown,
        )
        self._validate_errors_as_result = validate_errors_as_result

    def load(
        self,
        data: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]],
        *,
        many: Optional[bool] = None,
        partial: Optional[Union[bool, types.StrSequenceOrSet]] = None,
        unknown: Optional[str] = None,
    ):
        if self._validate_errors_as_result:
            return self.validate(data, many=many, partial=partial)
        return super().load(data, many=many, partial=partial, unknown=unknown)
