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

import pathlib

from flask.app import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase
from qhana_plugin_runner.util.plugins import plugin_identifier

_plugin_name = "cluster-scatter-visualization"
__version__ = "v0.0.3"
_identifier = plugin_identifier(_plugin_name, __version__)

VIS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="A visualization plugin for cluster scatter data.",
    template_folder="cluster_scatter_visualization_templates",
)

class ClusterScatterVisualization(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Visualizes cluster data in a scatter plot."
    tags = ["visualization"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

        # create folder for circuit images
        pathlib.Path(__file__).parent.absolute().joinpath("img").mkdir(
            parents=True, exist_ok=True
        )

    def get_api_blueprint(self):
        return VIS_BLP

    def get_requirements(self) -> str:
        return "pylatexenc~=2.10\nqiskit~=0.43\nkaleido~=0.2.1\ndash~=2.18.1"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
