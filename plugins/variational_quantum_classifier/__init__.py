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
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "vqc"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


VQC_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="VQC plugin API",
)

qiskit_ml_version = "0.4.0"


class VQC(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "This plugin implements the Variational Quantum Classifier (VQC) by Qiskit [0]. It's currently using version "
        f"{qiskit_ml_version} of qiskit's machine learning library."
        "[0] [Qiskit documentation, Variational Quantum Classifier](https://qiskit.org/documentation/machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html#qiskit_machine_learning.algorithms.VQC)\n"
        "[1] [Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019).](https://doi.org/10.1038/s41586-019-0980-2)"
    )
    tags = ["classification", "quantum"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return VQC_BLP

    def get_requirements(self) -> str:
        return f"qiskit~=0.43\nqiskit-machine-learning~={qiskit_ml_version}\nscikit-learn~=1.1"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
