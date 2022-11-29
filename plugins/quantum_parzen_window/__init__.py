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
    description = (
        "This plugin implements a quantum parzen window. A parzen window labels an unlabeled data point via a majority "
        "vote of all labeled points that are at most a certain distance away from the unlabeled point."
        "The Plugin implements the algorithm by Ruan et al. [0].\n\n"
        "Source:\n"
        "[0] [Ruan, Y., Xue, X., Liu, H. et al. Quantum Algorithm for K-Nearest Neighbors Classification Based on the Metric of Hamming Distance. Int J Theor Phys 56, 3496–3507 (2017).](https://doi.org/10.1007/s10773-017-3514-4)\n"
    )
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
