from typing import ClassVar, Optional

from celery.utils.log import get_task_logger
from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier


from .config import WorkflowPluginConfig, get_config

TASK_LOGGER = get_task_logger(__name__)

_plugin_name = "workflows"
__version__ = "v0.6.0"
_identifier = plugin_identifier(_plugin_name, __version__)

WORKFLOWS_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="BPMN workflows plugin API.",
    template_folder="templates",
)


class Workflows(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "WIP Workflow plugin executing workflows using the camunda bpmn engine."
    tags = ["workflow", "bpmn"]

    instance: ClassVar["Workflows"]

    config: WorkflowPluginConfig

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

        conf = get_config(app)

        self.config = conf

    def get_api_blueprint(self):
        return WORKFLOWS_BLP


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes  # noqa
    from .watchers.external import schedule  # noqa
    from . import management  # noqa
except ImportError as e:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
