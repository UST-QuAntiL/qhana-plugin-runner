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


_plugin_name = "max_cut"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


MaxCut_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Max Cut plugin API",
)


class MaxCut(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Clusters data with the max cut algorithm"

    tags = []

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return MaxCut_BLP

    def get_requirements(self) -> str:
        return "qiskit~=0.27\nqiskit-optimization~=0.5.0\ngit+https://github.com/pandrey-fr/maxcut.git@7ab0cf4a8131333ef6e67f23c4e43c190837c315\nplotly~=5.6.0\npandas~=1.5.0"


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
