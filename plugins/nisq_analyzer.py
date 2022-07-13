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

from http import HTTPStatus
from typing import Mapping, Optional

from flask import abort
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "nisq-analyzer"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


NISQ_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="NISQ Analyzer Plugin.",
)


@NISQ_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @NISQ_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @NISQ_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = NisqAnalyzer.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{NISQ_BLP.name}.PluginsView"),
                ui_href=url_for(f"{NISQ_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[],
                data_output=[],
            ),
            tags=NisqAnalyzer.instance.tags,
        )


@NISQ_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the NISQ Analyzer plugin."""

    @NISQ_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the NISQ Analyzer plugin."
    )
    @NISQ_BLP.arguments(
        FrontendFormBaseSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @NISQ_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    def render(self, data: Mapping, errors: dict):
        plugin = NisqAnalyzer.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return Response(
            render_template(
                "nisq_analyzer.html",
                name=plugin.name,
                version=plugin.version,
                values=data,
                errors=errors
            )
        )


class NisqAnalyzer(QHAnaPluginBase):

    name = _plugin_name
    version = __version__
    description = "Provides the NISQ Analyzer UI."
    tags = ["nisq-analyzer"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return NISQ_BLP
