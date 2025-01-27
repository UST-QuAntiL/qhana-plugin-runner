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

_plugin_name = "quantum-k-nearest-neighbours"
__version__ = "v0.2.1"
_identifier = plugin_identifier(_plugin_name, __version__)


QKNN_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Quantum k nearest neighbours plugin API.",
    template_folder="templates",
)


class QKNN(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "This plugin implements quantum k nearest neighbours algorithms. Given a set of already labeled data "
        "and an integer k, a new data point is labeled by a majority vote of the k nearest training points.\n"
        "The entity points should be saved in the [entity/vector](https://qhana-plugin-runner.readthedocs.io/en/latest/data-formats/examples/entities.html#entity-vector) format "
        "and labels in the [entity/label](https://qhana-plugin-runner.readthedocs.io/en/latest/data-formats/examples/entities.html#entity-label) format. "
        "Both may be stored in either a csv or a json file. Both can be generated with the ``data-creator`` plugin.\n\n"
        "Source:\n"
        "[0] [Schuld, M., Sinayskiy, I., Petruccione, F. (2014). Quantum Computing for Pattern Classification. In: Pham, DN., Park, SB. (eds) PRICAI 2014: Trends in Artificial Intelligence. PRICAI 2014. Lecture Notes in Computer Science(), vol 8862. Springer, Cham.](https://doi.org/10.1007/978-3-319-13560-1_17)\n"
        "[1] [Basheer, Afrad and Afham, A. and Goyal, Sandeep K. (2020). Quantum k-nearest neighbors algorithm. In arXiv.](https://doi.org/10.48550/arXiv.2003.09187)\n"
        "[2] [Ruan, Y., Xue, X., Liu, H. et al. Quantum Algorithm for K-Nearest Neighbors Classification Based on the Metric of Hamming Distance. Int J Theor Phys 56, 3496–3507 (2017).](https://doi.org/10.1007/s10773-017-3514-4)\n"
    )
    tags = ["QML", "clustering", "quantum", "supervised-learning"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QKNN_BLP

    def get_requirements(self) -> str:
        return "qiskit~=0.43\npennylane~=0.30\npennylane-qiskit~=0.30\nscikit-learn~=1.1\nmuid~=0.5.3"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
