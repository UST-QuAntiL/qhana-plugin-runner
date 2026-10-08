from qhana_plugin_runner.api.util import SecurityBlueprint

from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_name = "low-code-modeler"
_version = "v0.0.0"

LCM_BLP = SecurityBlueprint(
    plugin_identifier(_name, _version),
    __name__,
    description="low code modeler api",
    template_folder="templates",
)


class LowCodeModeler(QHAnaPluginBase):
    name = _name
    version = _version
    description = "low code modeler plugin"
    tags = ["low-code-modeler"]

    def get_api_blueprint(self):
        return LCM_BLP
