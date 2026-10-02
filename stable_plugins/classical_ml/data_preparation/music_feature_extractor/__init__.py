# Copyright 2026 QHAna plugin runner contributors.
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

_plugin_name = "music-feature-extractor"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


MusicFeatureExtractor_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="Music feature extractor plugin API.",
    template_folder="templates",
)


class MusicFeatureExtractorPlugin(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = (
        "Extracts stable feature vectors from MusicXML, MXL, and MIDI sources. "
        "The vector output follows the [entity/vector](https://qhana-plugin-runner.readthedocs.io/en/latest/data-formats/examples/entities.html#entity-vector) format."
    )
    tags = ["music", "feature-extraction", "data-preparation"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return MusicFeatureExtractor_BLP

    def get_requirements(self) -> str:
        return "music21~=9.1"


try:
    from . import routes
except ImportError:
    # Import can fail during plugin dependency installation.
    pass
