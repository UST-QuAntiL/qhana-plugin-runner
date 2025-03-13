from time import time
from typing import ClassVar, Optional

from flask import Blueprint, Flask
from kombu.exceptions import ConnectionError

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.db.models.virtual_plugins import PluginState
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

from .config import (
    CONFIG_KEY,
    get_config_from_app,
    get_config_from_registry,
    load_config_from_env,
    postprocess_config,
)

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

    def init_app(self, app: Flask):
        super().init_app(app)
        load_config_from_env(app)

    def get_config(self) -> dict:
        # load default config from app config
        config = get_config_from_app(self.app)
        saved_config = PluginState.get_value(self.name, CONFIG_KEY, None)
        assert saved_config is None or isinstance(saved_config, dict)
        last_updated = saved_config.get("_updated", 0) if saved_config else 0
        now = time()
        assert isinstance(last_updated, (int, float))
        if last_updated < (now - 3600):
            saved_config = get_config_from_registry(self.app)
            if saved_config:
                saved_config["_updated"] = time()
                PluginState.set_value(self.name, CONFIG_KEY, saved_config, commit=True)
            else:
                PluginState.delete_value(self.name, CONFIG_KEY)
        elif last_updated < (now - 60):
            try:
                update_config.apply_async(expires=30, retry=False)
            except NameError:
                pass  # called too early, name not yet defined
            except ConnectionError:
                pass  # redis is not available, do nothing
        config.update(saved_config)

        return postprocess_config(config)

    def get_api_blueprint(self):
        if not self._blueprint:
            self._blueprint = WF_EDITOR_BLP
            return WF_EDITOR_BLP
        return self._blueprint


from .tasks import update_config
