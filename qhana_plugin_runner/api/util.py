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
import http
import re
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union
from urllib.parse import urlparse

import marshmallow as ma
from flask_smorest import Blueprint
from flask_smorest.utils import remove_none
from marshmallow import types
from marshmallow.exceptions import ValidationError
from marshmallow.validate import URL as UrlValidator

from .jwt_helper import JWTMixin


class HtmlResponseMixin:
    """Extend Blueprint to add html response documentation."""

    def html_response(
        self,
        status_code: Union[int, str, http.HTTPStatus],
        *,
        description: str = None,
        example=None,
        examples=None,
        headers: dict = None,
    ):
        """Decorator documenting a html response adapted from :py:func:`ResponseMixin.alt_response`.

        Args:
            status_code (int|str|HTTPStatus): HTTP status code.
            description (str, optional): Description of the response. Defaults to None.
            example (dict, optional): Example of response message. Defaults to None.
            examples (dict, optional): Examples of response message. Defaults to None.
            headers (dict, optional): Headers returned by the response. Defaults to None.
        """
        if description is None:
            description = http.HTTPStatus(int(status_code)).phrase
        resp_doc = remove_none(
            {
                "content": {"text/html": {"schema": {"type": "string"}}},
                "description": description,
                "example": example,
                "examples": examples,
                "headers": headers,
            }
        )

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            # Store doc in wrapper function
            # The deepcopy avoids modifying the wrapped function doc
            wrapper._apidoc = deepcopy(getattr(wrapper, "_apidoc", {}))
            wrapper._apidoc.setdefault("response", {}).setdefault("responses", {})[
                status_code
            ] = resp_doc

            return wrapper

        return decorator


class SecurityBlueprint(Blueprint, JWTMixin, HtmlResponseMixin):
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


class FileUrlValidator(UrlValidator):
    """Extension of the URL validator that can handle file and data URLs"""

    default_schemes = {"http", "https", "ftp", "ftps", "file", "data"}

    _data_url_regex = re.compile(
        r"data:(?P<mime>[\w/\-\.+]+)(?P<charset>;charset=[\w/\-\.+]+)?(?P<encoding>;base64)?,.*"
    )

    def __call__(self, value: str) -> str:
        if value and value.startswith("file://"):
            return self._validate_file_url(value)
        if value and value.startswith("data:"):
            return self._validate_data_url(value)

        super().__init__(require_tld=False)

        return super().__call__(value)

    def _validate_file_url(self, value: str) -> str:
        result = urlparse(value)
        if result.netloc:
            if result.netloc != "localhost":
                raise ValidationError(self._format_error(value))
        else:
            pass  # just hope that url is correct...
        return value

    def _validate_data_url(self, value: str) -> str:
        if not self._data_url_regex.search(value):
            raise ValidationError(self._format_error(value))
        return value


class DataUrlValidator(UrlValidator):
    """Extension of the URL validator that can handle file and data URLs"""

    default_schemes = {"http", "https", "ftp", "ftps", "data"}

    _data_url_regex = re.compile(
        r"data:(?P<mime>[\w/\-\.+]+)(?P<charset>;charset=[\w/\-\.+]+)?(?P<encoding>;base64)?,.*"
    )

    def __call__(self, value: str) -> str:
        if value and value.startswith("data:"):
            return self._validate_data_url(value)

        super().__init__(require_tld=False)

        return super().__call__(value)

    def _validate_file_url(self, value: str) -> str:
        result = urlparse(value)
        if result.netloc:
            if result.netloc != "localhost":
                raise ValidationError(self._format_error(value))
        else:
            pass  # just hope that url is correct...
        return value

    def _validate_data_url(self, value: str) -> str:
        if not self._data_url_regex.search(value):
            raise ValidationError(self._format_error(value))
        return value


class FileUrl(ma.fields.Url):
    """Extension of the URL field that can handle file and data URLs"""

    def __init__(
        self,
        *,
        relative: bool = False,
        schemes: Optional[types.StrSequenceOrSet] = None,
        require_tld: bool = True,
        data_input_type: Optional[str] = None,
        data_content_types: Optional[Union[Sequence[str], str]] = None,
        required: bool = False,
        allow_none: bool = False,
        **kwargs,
    ):
        super().__init__(
            relative=relative,
            schemes=schemes,
            require_tld=require_tld,
            allow_none=allow_none,
            required=required,
            **kwargs,
        )
        if data_input_type is not None:
            self.metadata["data_input_type"] = data_input_type
        if data_content_types:
            self.metadata["data_content_types"] = data_content_types
        self.validators[0] = FileUrlValidator(
            relative=self.relative,
            schemes=schemes,
            require_tld=self.require_tld,
            error=self.error_messages["invalid"],
        )

    def deserialize(
        self,
        value: Any,
        attr: Optional[str] = None,
        data: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        # treat empty string as None
        if value == "":
            value = None
        return super().deserialize(value, attr, data, **kwargs)


class PluginUrl(ma.fields.URL):
    """Extension of the URL field that can handle plugin URLs"""

    def __init__(
        self,
        *,
        relative: bool = False,
        schemes: Optional[types.StrSequenceOrSet] = None,
        require_tld: bool = True,
        plugin_tags: Optional[Union[Sequence[str], str]] = None,
        plugin_name: Optional[str] = None,
        plugin_version: Optional[str] = None,
        required: bool = False,
        allow_none: bool = False,
        **kwargs,
    ):
        super().__init__(
            relative=relative,
            schemes=schemes,
            require_tld=require_tld,
            allow_none=allow_none,
            required=required,
            **kwargs,
        )
        if plugin_tags:
            self.metadata["plugin_tags"] = plugin_tags
        if plugin_name:
            self.metadata["plugin_name"] = plugin_name
        if plugin_version:
            self.metadata["plugin_version"] = plugin_version
        self.validators[0] = DataUrlValidator(
            relative=self.relative,
            schemes=schemes,
            require_tld=self.require_tld,
            error=self.error_messages["invalid"],
        )

    def deserialize(
        self,
        value: Any,
        attr: Optional[str] = None,
        data: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        # treat empty string as None
        if value == "":
            value = None
        return super().deserialize(value, attr, data, **kwargs)
