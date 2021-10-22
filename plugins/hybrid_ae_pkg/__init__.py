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
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


HA_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="Hybrid Autoencoder plugin API.",
    template_folder="hybrid_ae_templates",
)


class HybridAutoencoderPlugin(QHAnaPluginBase):
    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return HA_BLP

    def get_requirements(self) -> str:
        # return "git+ssh://git@github.com/UST-QuAntiL/MuseEmbeddings.git@6cc2f18fdd6b9483d5aaa68d12f8e01cb6329dde#egg=hybrid_autoencoders"
        # TODO: remove dependency on the MuseEmbeddings project
        return ""


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    import plugins.hybrid_ae_pkg.routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
