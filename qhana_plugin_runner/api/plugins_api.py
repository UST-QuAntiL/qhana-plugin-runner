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

"""Module containing the root endpoint of the v1 API."""

from dataclasses import dataclass
from http import HTTPStatus
from typing import List

import marshmallow as ma
from flask.helpers import url_for
from flask.views import MethodView

from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase

PLUGINS_API = SmorestBlueprint(
    "plugins-api", "Plugins API", description="Api to request a list of loaded plugins.", url_prefix="/plugins"
)


@dataclass()
class PluginData:
    name: str
    version: str
    identifier: str


@dataclass()
class PluginCollectionData:
    plugins: List[PluginData]


class PluginSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class PluginCollectionSchema(MaBaseSchema):
    plugins = ma.fields.List(ma.fields.Nested(PluginSchema()))


@PLUGINS_API.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PLUGINS_API.response(HTTPStatus.OK, PluginCollectionSchema())
    def get(self):
        """Get all loaded plugins."""
        plugins = QHAnaPluginBase.get_plugins().values()
        return PluginCollectionData(plugins=[PluginData(p.name, p.version, p.identifier) for p in plugins])
