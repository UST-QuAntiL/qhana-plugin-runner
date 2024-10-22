# Copyright 2024 QHAna plugin runner contributors.
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

_plugin_name = "muse-for-music-loader"
__version__ = "v0.1.1"
_identifier = plugin_identifier(_plugin_name, __version__)


M4MLoader_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="MUSE4Music Loader plugin API",
    template_folder="templates",
)


class M4MLoaderPlugin(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Load data from a MUSE4Music instance."

    tags = ["MUSE4Music", "data-loading"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return M4MLoader_BLP

    # def get_requirements(self) -> str:
    #     return ""


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and the Plugin are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
