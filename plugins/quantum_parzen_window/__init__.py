from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "quantum-parzen-window"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


QParzenWindow_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Quantum parzen window plugin API.",
)


class QParzenWindow(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "parzen window algorithms that can run on quantum computers."
    tags = ["parzen-window", "supervised-learning", "quantum"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QParzenWindow_BLP

    def get_requirements(self) -> str:
        return "pennylane~=0.16\npennylane-qiskit~=0.16"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
