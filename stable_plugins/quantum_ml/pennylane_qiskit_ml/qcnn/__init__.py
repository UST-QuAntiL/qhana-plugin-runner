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


_plugin_name = "quantum-cnn"
__version__ = "v0.1.1"
_identifier = plugin_identifier(_plugin_name, __version__)


QCNN_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Quantum CNN plugin API",
    template_folder="templates",
)


class QCNN(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Labels data with the help of a quantum convolutional neural network"

    tags = ["QML", "classification", "quantum", "neural-network"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QCNN_BLP

    def get_requirements(self) -> str:
        return "qiskit~=0.43\npennylane~=0.30\npennylane-qiskit~=0.30\nscikit-learn~=1.1\ntorch~=2.0.1\nplotly~=5.18.0\npandas~=1.5.0"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
