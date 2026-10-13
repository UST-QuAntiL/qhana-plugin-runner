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

import textwrap
from typing import Optional

from flask.app import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "attribute-distance-aggregator"
__version__ = "v0.2.0"
_identifier = plugin_identifier(_plugin_name, __version__)


AGGREGATOR_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Attribute distance aggregator plugin API.",
)


class AttributeAggregator(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = textwrap.dedent(
        r"""
    Aggregates element distances to attribute distances for a list of entities.
    \
    For a taxonomy attribute the elements of an entity are its attribute values, and the
    attribute distance is the Sym Max Mean of the element distances between the two entities.
    \
    A numeric attribute (attribute metadata description `number`, `integer`, `int`, `float`
    or `double`) has one element per entity, keyed by the entity ID, as produced by the
    mapping distances plugin. Sym Max Mean over the two single elements returns their
    element distance unchanged.
    \
    A pair where one entity has no element for the attribute gets the distance `null`.
    """
    ).strip()
    tags = ["preprocessing", "distance-calculation"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return AGGREGATOR_BLP


try:
    from . import routes  # noqa: F401,E402
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
