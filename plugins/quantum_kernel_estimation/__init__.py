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

_plugin_name = "quantum-kernel-estimation"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


QKE_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Quantum-Kernel-Estimation plugin API",
)


class QKE(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "This plugin produces the matrix of a quantum kernel. Since this depends on the expected values of "
        "the quantum circuit, we can only estimate it and therefore call it Quantum Kernel Estimation. "
        "The Plugin implements the kernels by Havlíček et al [0] and Suzuki et al [1].\n\n"
        "Source:\n"
        "[0] [Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019).](https://doi.org/10.1038/s41586-019-0980-2)\n"
        "[1] [Suzuki, Y., Yano, H., Gao, Q. et al. Analysis and synthesis of feature map for kernel-based quantum classifier. Quantum Mach. Intell. 2, 9 (2020).](https://doi.org/10.1007/s42484-020-00020-y)"
    )
    tags = ["kernel", "mapping"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QKE_BLP

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
