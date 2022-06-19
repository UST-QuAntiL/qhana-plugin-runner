from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "qnn"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


QNN_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="QNN plugin API",
)


class QNN(QHAnaPluginBase):
    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QNN_BLP

    def get_requirements(self) -> str:
        return "numpy~=1.22.2\nmatplotlib~=3.5.1\nPennyLane~=0.16\nscikit-learn~=0.24.2\ntorch~=1.11.0"
        # return "numpy~=1.22.2\nmatplotlib~=3.5.1\nPennyLane~=0.7.0\nscikit-learn~=0.24.2\ntorch~=1.11.0"  # TODO? # after specifying here "poetry run flask install"
        # return "numpy~=1.22.2\nmatplotlib~=3.5.1\npennylane~=0.16\nscikit-learn~=0.24.2"  # TODO? # after specifying here "poetry run flask install"
        # \ntorch~=1.11.0


# qiskit~=0.27

try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    import plugins.qnn.routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
