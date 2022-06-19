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

"""Module containing the endpoints related to plugins."""

from dataclasses import dataclass
from http import HTTPStatus
from typing import List, Optional

import marshmallow as ma
from flask.helpers import url_for
from flask.views import MethodView
from flask_smorest import abort
from werkzeug.utils import redirect

from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase

PLUGINS_API = SmorestBlueprint(
    "plugins-api",
    __name__,
    description="Api to request a list of loaded plugins.",
    url_prefix="/plugins",
)


@dataclass()
class PluginData:
    name: str
    version: str
    identifier: str
    api_root: Optional[str]
    description: str
    tags: List[str]


@dataclass()
class PluginCollectionData:
    plugins: List[PluginData]


class PluginSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)
    api_root = ma.fields.Url(required=False, allow_none=False, dump_only=True)
    description = ma.fields.String(required=True, allow_none=False, dump_only=True)
    tags = ma.fields.List(ma.fields.String(), required=True, dump_only=True)


class PluginCollectionSchema(MaBaseSchema):
    plugins = ma.fields.List(ma.fields.Nested(PluginSchema()))


@PLUGINS_API.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PLUGINS_API.response(HTTPStatus.OK, PluginCollectionSchema())
    def get(self):
        """Get all loaded plugins."""
        plugins = sorted(
            (p for p in QHAnaPluginBase.get_plugins().values() if p.has_api),
            key=lambda p: (p.name, p.parsed_version),
        )
        return PluginCollectionData(
            plugins=[
                PluginData(
                    p.name,
                    p.version,
                    p.identifier,
                    url_for(
                        "plugins-api.PluginView", plugin=p.identifier, _external=True
                    ),
                    p.description,
                    p.tags,
                )
                for p in plugins
            ]
        )


@PLUGINS_API.route("/<string:plugin>/")
class PluginView(MethodView):
    """Generic fallback plugins view."""

    @PLUGINS_API.response(HTTPStatus.TEMPORARY_REDIRECT)
    def get(self, plugin: str):
        """Redirect to the newest version of a plugin."""
        plugins = QHAnaPluginBase.get_plugins()
        if plugin in plugins:
            abort(
                HTTPStatus.NOT_FOUND, message="The plugin does not provide a blueprint."
            )
        found_plugin: Optional[QHAnaPluginBase] = None
        for p in plugins.values():
            if p.name != plugin:
                continue
            if found_plugin is None or found_plugin.parsed_version < p.parsed_version:
                found_plugin = p

        if found_plugin is None:
            abort(HTTPStatus.NOT_FOUND, message="No plugin registered with that name.")

        try:
            found_plugin.get_api_blueprint()
        except NotImplementedError:
            abort(HTTPStatus.NOT_FOUND, message="No plugin registered with that name.")

        return redirect(url_for("plugins-api.PluginView", plugin=found_plugin.identifier))
