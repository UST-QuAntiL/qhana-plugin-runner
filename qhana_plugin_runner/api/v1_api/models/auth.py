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

"""Module containing all API schemas for the authentication API."""

import marshmallow as ma
from ...util import MaBaseSchema

__all__ = [
    "AuthRootSchema",
    "LoginPostSchema",
    "LoginTokensSchema",
    "AccessTokenSchema",
    "UserSchema",
]


class AuthRootSchema(MaBaseSchema):
    login = ma.fields.Url(required=True, allow_none=False, dump_only=True)
    refresh = ma.fields.Url(required=True, allow_none=False, dump_only=True)
    whoami = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class LoginPostSchema(MaBaseSchema):
    username = ma.fields.String(required=True, allow_none=False)
    password = ma.fields.String(required=True, allow_none=False, load_only=True)


class LoginTokensSchema(MaBaseSchema):
    access_token = ma.fields.String(required=True, allow_none=False, dump_only=True)
    refresh_token = ma.fields.String(required=True, allow_none=False, dump_only=True)


class AccessTokenSchema(MaBaseSchema):
    access_token = ma.fields.String(required=True, allow_none=False, dump_only=True)


class UserSchema(MaBaseSchema):
    username = ma.fields.String(required=True, allow_none=False)
