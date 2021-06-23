from http import HTTPStatus
from typing import Optional

import marshmallow as ma
from flask.app import Flask
from flask.views import MethodView

from qhana_plugin_runner.api.util import MaBaseSchema, SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "hello-world"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

HELLO_BLP = SecurityBlueprint(_identifier, __name__, description="Demo plugin API.")


class DemoResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    version = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


@HELLO_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @HELLO_BLP.response(HTTPStatus.OK, DemoResponseSchema())
    @HELLO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Demo endpoint returning the plugin metadata."""
        return {
            "name": HelloWorld.instance.name,
            "version": HelloWorld.instance.version,
            "identifier": HelloWorld.instance.identifier,
        }


class HelloWorld(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)
        print("\nInitialized hello world plugin.\n")

    def get_api_blueprint(self):
        return HELLO_BLP
