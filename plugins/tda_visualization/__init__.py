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

_plugin_name = "tda-visualization"
__version__ = "v1.0.0"
_identifier = plugin_identifier(_plugin_name, __version__)

TDA_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="A Topological Data Analysis (TDA) visualization plugin that creates a persistence diagram using the provided data."
    + " When an Entity Point URL is provided, a simple persistence diagram plot will be created.",
    template_folder="tda_visualization_templates",
)


class TDAVisualization(QHAnaPluginBase):
    name = "tda-visualization"
    version = __version__
    description = (
        "A visualization plugin that creates a persistence diagram using the provided data."
        + " When an Entity Point URL is provided, a simple persistence diagram plot will be created."
    )
    tags = ["visualization", "tda", "persistence"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return TDA_BLP

    def get_requirements(self) -> str:
        return "scikit-tda~=1.1\nnumpy~=1.26\nplotly~=5.18.0"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
    from . import tasks
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
