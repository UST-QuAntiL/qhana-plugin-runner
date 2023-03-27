from os import environ
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from celery.utils.log import get_task_logger
from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

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

    config: Dict[
        str, Any
    ]  # FIXME use full config from existing workflows plugin (fix in __init__)

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

        app_config = app.config if app else {}

        workflow_folder = Path(
            app_config.get(
                "WORKFLOW_FOLDER",
                environ.get("WORKFLOW_FOLDER", "./workflows"),
            )
        )
        if not workflow_folder.is_absolute() and app is not None:
            workflow_folder = Path(app.instance_path) / workflow_folder
            workflow_folder = workflow_folder.resolve()

        camunda_url = app_config.get(
            "CAMUNDA_API_URL",
            environ.get("CAMUNDA_API_URL", "http://localhost:8080/engine-rest"),
        )

        default_timout: str = app_config.get(
            "REQUEST_TIMEOUT",
            environ.get("REQUEST_TIMEOUT", str(5 * 60)),
        )
        timout_int = 5 * 60
        if default_timout.isdigit():
            timout_int = int(default_timout)

        conf = {
            "WORKFLOW_FOLDER": workflow_folder,
            "CAMUNDA_BASE_URL": camunda_url,
            "request_timeout": timout_int,
            # TODO
        }

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
