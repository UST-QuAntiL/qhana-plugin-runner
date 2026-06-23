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

from typing import ClassVar, Optional

from flask import Blueprint, Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_name = "mapping-distances"
_version = "v0.1.0"
_identifier = plugin_identifier(_name, _version)


MAPPING_DISTANCES_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description="Taxanomy mapping to distances plugin API.",
)


class MappingDistances(QHAnaPluginBase):
    name = _name
    version = _version
    description = "A plugin to create element distances for taxanomy mappings."
    tags = ["preprocessing", "distance-calculation"]

    instance: ClassVar["MappingDistances"]

    _blueprint: Optional[Blueprint] = None

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return MAPPING_DISTANCES_BLP

    def get_requirements(self) -> str:
        return "muse-for-music-loader~=1.1.0"


try:
    from . import routes  # noqa: F401,E402
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
