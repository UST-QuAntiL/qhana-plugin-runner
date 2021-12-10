from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "manual-classification"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)

MANUAL_CLASSIFICATION_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Manual Classification API.",
    template_folder="manual_classification_templates",
)


class ManualClassification(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return MANUAL_CLASSIFICATION_BLP

    def get_requirements(self) -> str:
        return ""


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    import plugins.manual_classification.routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
