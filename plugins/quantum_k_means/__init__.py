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
    description = (
        "This plugin groups the data into different clusters, with the help of quantum algorithms.\n"
        "Currently there are four implemented algorithms. Destructive interference and negative rotation are from [0], "
        "positive correlation is from [1] and state preparation is from a previous colleague.\n\n"
        "Source:\n"
        '[0] [S. Khan and A. Awan and G. Vall-Llosera. K-Means Clustering on Noisy Intermediate Scale Quantum Computers.arXiv.](https://doi.org/10.48550/ARXIV.1909.12183)\n'
        "[1] <https://towardsdatascience.com/quantum-machine-learning-distance-estimation-for-k-means-clustering-26bccfbfcc76>"
    )
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
