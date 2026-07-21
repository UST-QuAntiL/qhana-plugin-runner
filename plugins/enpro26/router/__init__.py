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

_plugin_name = "router"
__version__ = "v0.1.4"
_identifier = plugin_identifier(_plugin_name, __version__)
_description = (  # TODO
    "Routes entities and separates tree from non-tree taxonomies. "
    "Takes Muse4Music data and provides different routing options for the entities and taxonomies."
)

ROUTER_BLP = SecurityBlueprint(
    _identifier,
    __name__,
    description=_description,
    template_folder="templates",
)


class Router(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = _description
    tags = ["preprocessing", "routing"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return ROUTER_BLP


try:
    from . import routes  # noqa: F401,E402
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
