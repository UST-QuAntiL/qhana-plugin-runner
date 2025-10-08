from qhana_plugin_runner.api.util import SecurityBlueprint

from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_name = "pattern-atlas"
_version = "v0.0.0"

PA_BLP = SecurityBlueprint(
    plugin_identifier(_name, _version),
    __name__,
    description="pattern atlas api",
    template_folder="pattern_atlas_dynamic/templates",
)


class PatternAtlas(QHAnaPluginBase):
    name = _name
    version = _version
    description = "pattern atlas plugin"
    tags = ["pattern-atlas"]

    def get_api_blueprint(self): 
        from . import routes
        return PA_BLP

    def get_requirements(self):
        return """\
httpx~=0.25.0
jinja2~=3.1.2
mistune~=3.0.0
markupsafe~=2.1.3
"""
