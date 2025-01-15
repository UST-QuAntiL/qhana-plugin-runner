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

_plugin_name = "confusion_matrix"
__version__ = "v0.4.0"
_identifier = plugin_identifier(_plugin_name, __version__)

VIS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="A visualization plugin that creates a confusion matrix using the provided data."
    + " Accepts two cluster URLs as inputs an outputs an HTML table showing the matrix."
    + " If desired the matrix can be optimized, trying to maximize the amount of true positives,"
    + " by reordering the columns. The new column order will be shown at the bottom.",
    template_folder="confusion_matrix_templates",
)


class ConfusionMatrixVisualization(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "A visualization plugin that creates a confusion matrix using the provided data."
        + " Accepts two cluster URLs as inputs an outputs an HTML table showing the matrix."
        + " If desired the matrix can be optimized, trying to maximize the amount of true positives,"
        + " by reordering the columns. The new column order will be shown at the bottom."
    )
    tags = ["visualization", "cluster", "confusion matrix"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return VIS_BLP

    def get_requirements(self) -> str:
        return "pylatexenc~=2.10\nscipy~=1.15.0\nnumpy~=2.1.3"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
