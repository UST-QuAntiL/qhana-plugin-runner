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

_plugin_name = "histogram-visualization"
__version__ = "v1.0.0"
_identifier = plugin_identifier(_plugin_name, __version__)

VIS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="A visualization plugin for creating Historgrams using the counts of different labels."
    + "The labels are shown on the x Axis and the counts on the y Axis.",
    template_folder="histogram_visualization_templates",
)


class HistogramVisualization(QHAnaPluginBase):
    name = "Histogram Visualization"
    version = __version__
    description = (
        "A visualization plugin for creating Historgrams using the counts of different labels."
        + "The labels are shown on the x Axis and the counts on the y Axis."
    )
    tags = ["visualization", "histogram"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return VIS_BLP

    def get_requirements(self) -> str:
        return "pylatexenc~=2.10\nkaleido~=0.2.1\n"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
