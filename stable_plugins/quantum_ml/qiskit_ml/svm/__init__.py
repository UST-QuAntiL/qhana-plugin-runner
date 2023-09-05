# Copyright 2023 QHAna plugin runner contributors.
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
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier


_plugin_name = "svm"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


SVM_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="SVM plugin API",
    template_folder="templates",
)

sklearn_version = "1.1"


class SVM(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "Classifies data with a support vector machine. This plugin uses the implementation of "
        f"scikit-learn {sklearn_version} [0]. The quantum kernels are from Qiskit [1] and the data maps "
        f"are from Havlíček et al. [2] and Suzuki et al. [3].\n\n"
        "Source:\n"
        "[0] [https://scikit-learn.org/1.1/modules/svm.html#svm](https://scikit-learn.org/1.1/modules/svm.html#svm)\n"
        "[1] Qiskit's quantum kernels [ZFeatureMap](https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZFeatureMap.html), "
        "[ZZFeatureMap](https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZZFeatureMap.html) and "
        "[PauliFeatureMap](https://qiskit.org/documentation/stubs/qiskit.circuit.library.PauliFeatureMap.html)\n"
        "[2] [Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019).](https://doi.org/10.1038/s41586-019-0980-2)\n"
        "[3] [Suzuki, Y., Yano, H., Gao, Q. et al. Analysis and synthesis of feature map for kernel-based quantum classifier. Quantum Mach. Intell. 2, 9 (2020).](https://doi.org/10.1007/s42484-020-00020-y)"
    )

    tags = ["quantum", "classical", "supervised"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return SVM_BLP

    def get_requirements(self) -> str:
        return f"qiskit~=0.43\nqiskit-machine-learning~=0.4.0\nscikit-learn~={sklearn_version}\nplotly~=5.6.0\npandas~=1.5.0"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
