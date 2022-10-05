# Copyright 2022 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


qiskit_version = "0.27"
qiskit_ml_version = "0.4.0"


class QiskitQKE(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "Produces a kernel matrix from a quantum kernel. "
        "Specifically qiskit's feature maps are used, combined with qiskit_machine_learning.kernels.QuantumKernel. These feature "
        "maps are ZFeatureMap, ZZFeatureMap, PauliFeatureMap from qiskit.circuit.library. These feature maps all use the proposed "
        f"kernel by Havlíček [0]. The following versions were used `qiskit~={qiskit_version}` and `qiskit-machine-learning~={qiskit_ml_version}`.\n\n"
        "Source:\n"
        "[0] [Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019).](https://doi.org/10.1038/s41586-019-0980-2)"
    )
    tags = ["kernel", "mapping"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QISKIT_QKE_BLP

    def get_requirements(self) -> str:
        return f"qiskit~={qiskit_version}\nqiskit-machine-learning~={qiskit_ml_version}"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
