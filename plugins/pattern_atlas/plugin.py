from qhana_plugin_runner.api.util import SecurityBlueprint

from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_name = "pattern-atlas"
_version = "v0.0.0"

PA_BLP = SecurityBlueprint(
    plugin_identifier(_name, _version),
    __name__,
    description="pattern atlas api",
    template_folder="templates",
)


class PatternAtlas(QHAnaPluginBase):
    name = _name
    version = _version
    description = "pattern atlas plugin"
    tags = ["pattern-atlas"]

    def get_api_blueprint(self):
        return PA_BLP
