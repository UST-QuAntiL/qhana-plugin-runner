from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier

from ..plugin import RESTConnector

REST_CONN_BLP = SecurityBlueprint(
    plugin_identifier(RESTConnector.name, RESTConnector.version),
    __name__,
    description="REST Connector Plugin API.",
    template_folder="templates",
)
