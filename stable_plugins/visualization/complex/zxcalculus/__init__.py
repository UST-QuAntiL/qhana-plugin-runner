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

import pathlib

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "zxcalculus"
__version__ = "v1.0.0"
_identifier = plugin_identifier(_plugin_name, __version__)


VIS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="A visualization plugin that visualizes a provided OpenQASM circuit in the ZX-Calculus."
    + " When a QASM Circuit URL is provided, a circuit in the ZX-Calculus will be created."
    + " When the Optimize Checkbox is checked, an additional circuit is generated."
    + " This circuit is optimized using the automatic optimization method provided by the pyzx package,"
    + " and will be displayed below the original circuit",
    template_folder="zxcalculus_visualization_templates",
)


class ZXCalculusVisualization(QHAnaPluginBase):
    name = "ZX-Calculus Visualization"
    version = __version__
    description = (
        "A visualization plugin that visualizes a provided OpenQASM circuit in the ZX-Calculus."
        + " When a QASM Circuit URL is provided, a circuit in the ZX-Calculus will be created."
        + " When the Optimize Checkbox is checked, an additional circuit is generated."
        + " This circuit is optimized using the automatic optimization method provided by the pyzx package,"
        + " and will be displayed below the original circuit"
    )
    tags = ["visualization", "zxcalculus", "circuit"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

        # create folder for circuit qasms
        pathlib.Path(__file__).parent.absolute().joinpath("files").mkdir(
            parents=True, exist_ok=True
        )

    def get_api_blueprint(self):
        return VIS_BLP

    def get_requirements(self) -> str:
        return "pylatexenc~=2.10\npyzx~=0.8.0\nmpld3~=0.5.10"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
