from typing import ClassVar, Optional

from flask import Blueprint, Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_name = "workflow-editor"
_version = "v0.1.0"


WF_EDITOR_BLP = SecurityBlueprint(
    plugin_identifier(_name, _version),
    __name__,
    description="Workflow Editor Plugin API.",
    template_folder="templates",
)


class WorkflowEditor(QHAnaPluginBase):
    name = _name
    version = _version
    description = "A plugin to edit BPMN files."
    tags = ["workflow", "camunda", "quantme"]

    instance: ClassVar["WorkflowEditor"]

    _blueprint: Optional[Blueprint] = None

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        if not self._blueprint:
            self._blueprint = WF_EDITOR_BLP
            return WF_EDITOR_BLP
        return self._blueprint
