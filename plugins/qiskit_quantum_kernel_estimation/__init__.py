from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "qiskit-quantum-kernel-estimation"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


QISKIT_QKE_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Qiskit Quantum-Kernel-Estimation plugin API",
)


class Qiskit_QKE(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Produces a kernel matrix from a quantum kernel. " \
                  "Specifically qiskit's feature maps are used, combined with qiskit_machine_learning.kernels.QuantumKernel. These feature " \
                  "maps are ZFeatureMap, ZZFeatureMap, PauliFeatureMap from qiskit.circuit.library. These feature maps all use the proposed " \
                  "kernel by Havlíček [0]. The following versions were used qiskit~=0.27 and qiskit-machine-learning~=0.4.0.\n\n" \
                  "Source:\n" \
                  "[0] Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019). <a href=\"https://doi.org/10.1038/s41586-019-0980-2\">https://doi.org/10.1038/s41586-019-0980-2</a>"
    tags = ["kernel", "mapping"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QISKIT_QKE_BLP

    def get_requirements(self) -> str:
            return "qiskit~=0.27\nqiskit-machine-learning~=0.4.0"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
