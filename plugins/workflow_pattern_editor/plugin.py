from typing import ClassVar, Optional

from flask import Blueprint, Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_name = "workflow-pattern-editor"
_version = "v0.1.0"

PATTERN_EDITOR_BLP = SecurityBlueprint(
    plugin_identifier(_name, _version),
    __name__,
    description="Workflow Pattern Editor Plugin API.",
    template_folder="templates",
)


class WorkflowPatternEditor(QHAnaPluginBase):
    name = _name
    version = _version
    description = "A plugin to edit workflow patterns and integrate with Pattern Atlas."
    tags = ["workflow", "patterns", "atlas"]

    instance: ClassVar["WorkflowPatternEditor"]

    _blueprint: Optional[Blueprint] = None

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        if not self._blueprint:
            self._blueprint = PATTERN_EDITOR_BLP
            # Import routes to register them with the blueprint
            from . import routes
            return PATTERN_EDITOR_BLP
        return self._blueprint