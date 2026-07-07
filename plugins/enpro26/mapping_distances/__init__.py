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
    description = (
        "A plugin to create pairwise element distances for taxanomy mappings.<br /><br />\n"
        "Returns the distance between all attribute mappings that are in the entity dataset "
        "according to a selected distance metric.<br />\n"
        "Returns the max float value if the vectors are empty, i.e. no mapping is assigned.<br />\n"
        "Throws an error if the mapping vectors do not have the same size.<br /><br />\n"
        "Different available metrics:<br />\n"
        "**Euclidean Distance:** Length of vector (L2 norm) between two vectors: "
        "$||a-b|| = \sqrt{\sum\limits_{i} (a_i - b_i)^2}$<br />\n"
        "**Manhattan Distance:** Sum of distances on each vector axis: $\sum\limits_{i} |a_i - b_i|$<br />\n"
        "**Chebyshev Distance:** Maximum distance on one axis: $\max(|a_1 - b_1|, \dots, |a_n - b_n|)$<br />\n"
        "**Cosine Distance:** 1 - angle between two vectors (value in [0, 2]): "
        "$1 - \cos(\\theta) = 1 - \\frac{a \cdot b}{||a||\cdot||b||}$"
    )
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
