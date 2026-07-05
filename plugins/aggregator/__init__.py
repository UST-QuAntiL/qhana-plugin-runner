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

from flask.app import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "attribute-distance-aggregator"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


AGGREGATOR_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Attribute distance aggregator plugin API.",
)


class AttributeAggregator(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Aggregates element distances to attribute distances."
    tags = ["preprocessing", "distance-calculation"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return AGGREGATOR_BLP

    def get_requirements(self) -> str:
        return "muid~=0.5.3"


try:
    from . import routes  # noqa: F401,E402
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
