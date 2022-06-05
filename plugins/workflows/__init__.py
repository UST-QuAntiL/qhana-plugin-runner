import yaml
from typing import Optional
from celery.utils.log import get_task_logger
from flask import Flask
from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

conf = yaml.safe_load(open("plugins/workflows/config.yml"))
TASK_LOGGER = get_task_logger(__name__)

_plugin_name = "workflows"
__version__ = "v0.5.1"
_identifier = plugin_identifier(_plugin_name, __version__)

WORKFLOWS_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="BPMN workflows plugin API.",
    template_folder="templates"
)


class Workflows(QHAnaPluginBase):
    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return WORKFLOWS_BLP

    def get_requirements(self) -> str:
        return "dataclasses-json~=0.5.7"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    import plugins.workflows.routes
    import plugins.workflows.watchers.external.schedule
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
