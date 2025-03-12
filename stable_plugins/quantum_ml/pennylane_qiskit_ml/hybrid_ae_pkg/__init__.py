# Copyright 2021 QHAna plugin runner contributors.
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

from flask.app import Flask
from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "hybrid-autoencoder"
__version__ = "v0.2.1"
_identifier = plugin_identifier(_plugin_name, __version__)


HA_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="Hybrid Autoencoder plugin API.",
    template_folder="templates",
)


class HybridAutoencoderPlugin(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "Reduces the dimensionality of a given dataset with a combination of classical and quantum neural networks.\n"
        "The entity points should be saved in the [entity/vector](https://qhana-plugin-runner.readthedocs.io/en/latest/data-formats/examples/entities.html#entity-vector) format "
        "and they may be stored in either a csv or a json file. The ``data-creator`` plugin can generate some entity points."
    )
    tags = ["QML", "preprocessing", "feature-engineering", "quantum"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return HA_BLP

    def get_requirements(self) -> str:
        return "qiskit~=0.43\npennylane~=0.30\npennylane-qiskit~=0.30\ntorch~=2.0.1\nmuid~=0.5.3"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
