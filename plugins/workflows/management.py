from os import environ
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from celery.utils.log import get_task_logger
from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

from .config import WorkflowPluginConfig, get_config

TASK_LOGGER = get_task_logger(__name__)

_plugin_name = "workflow-management"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

WORKFLOW_MGMNT_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="Plugin for managing BPMN workflows deployed in Camunda.",
    template_folder="templates",
)


class WorkflowManagement(
    QHAnaPluginBase
):  # FIXME this should replace the current workflows plugin completely
    name = _plugin_name
    version = __version__
    description = "Plugin for managing BPMN workflows deployed in Camunda."
    tags = ["workflow", "bpmn"]

    instance: ClassVar["WorkflowManagement"]

    config: WorkflowPluginConfig

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

        conf = get_config(app)

        self.config = conf

    def get_api_blueprint(self):
        return WORKFLOW_MGMNT_BLP


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import management_routes  # noqa
except ImportError as e:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
