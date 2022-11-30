import uuid
from json import loads
from os import environ
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from celery.utils.log import get_task_logger
from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

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

    config: Dict[str, Any]

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
        plugin_runner_url = app_config.get(
            "PLUGIN_RUNNER_URLS",
            [
                url.strip()
                for url in environ.get(
                    "PLUGIN_RUNNER_URLS", "http://localhost:5005;"
                ).split(";")
                if url.strip()
            ],
        )

        default_timout: str = app_config.get(
            "REQUEST_TIMEOUT",
            environ.get("REQUEST_TIMEOUT", str(5 * 60)),
        )
        timout_int = 5 * 60
        if default_timout.isdigit():
            timout_int = int(default_timout)

        max_parrallelism: str = app_config.get(
            "EXTERNAL_TASK_CONCURRENCY",
            environ.get("EXTERNAL_TASK_CONCURRENCY", str(10)),
        )
        max_parrallelism_int = 10
        if max_parrallelism.isdigit():
            max_parrallelism_int = int(max_parrallelism)

        worker_id: str = app_config.get(
            "CAMUNDA_WORKER_ID",
            environ.get("CAMUNDA_WORKER_ID", str(uuid.uuid4())),
        )

        workflow_config: Dict[str, float] = app_config.get("WORKFLOWs", {})
        env_workflow_config = environ.get("PLUGIN_WORKFLOWS", None)
        if env_workflow_config:
            workflow_config = loads(env_workflow_config)

        workflow_watcher_config: Dict[str, float] = app_config.get(
            "WORKFLOW_WATCHERS", {}
        )
        env_workflow_watcher_config = environ.get("PLUGIN_WORKFLOW_WATCHERS", None)
        if env_workflow_watcher_config:
            workflow_watcher_config = loads(env_workflow_watcher_config)

        conf = {
            "WORKFLOW_FOLDER": workflow_folder,
            "CAMUNDA_BASE_URL": camunda_url,
            "QHANA_PLUGIN_ENDPOINTS": plugin_runner_url,
            "worker_id": worker_id,
            "polling_rates": {"camunda_general": 5.0, "external_watcher": 5.0},
            "request_timeout": timout_int,
            "external_task_concurrency": max_parrallelism_int,
            "workflow_error_prefix": "qhana",
            "qhana_input": {
                "prefix": "qinput",
                "prefix_value_choice": "choice",
                "prefix_value_enum": "enum",
                "prefix_value_file_url": "file_url",
                "prefix_value_delimiter": "::",
                "mode_text": "plain",
                "mode_filename": "name",
                "mode_datatype": "dataType",
            },
            "workflow_out": {
                "camunda_user_task_name": "Workflow Return Variables",
                "prefix": "return",
            },
        }

        conf["polling_rates"].update(workflow_watcher_config)
        conf["qhana_input"].update(workflow_config.get("qhana_input", {}))
        conf["workflow_out"].update(workflow_config.get("workflow_out", {}))

        self.config = conf

    def get_api_blueprint(self):
        return WORKFLOWS_BLP


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes  # noqa
    from .watchers.external import schedule  # noqa
except ImportError as e:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
