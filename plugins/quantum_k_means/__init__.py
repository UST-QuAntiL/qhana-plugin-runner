from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "quantum-k-means"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


QKMEANS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Quantum k-means plugin API.",
)


class QKMeans(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "K-means algorithms that can run on quantum computers."
    tags = ["points-to-clusters", "k-means"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QKMEANS_BLP

    def get_requirements(self) -> str:
        return "qiskit~=0.27\npennylane~=0.16\npennylane-qiskit~=0.16"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
