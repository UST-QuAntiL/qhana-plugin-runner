from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "optimization-coordinator"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


OPTI_COORD_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Optimization coordinator API.",
    template_folder="hello_world_templates",
)


class OptimizationCoordinator(QHAnaPluginBase):

    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return OPTI_COORD_BLP


try:
    # It is important to import the routes **after** OPTI_COORD_BLP and OptimizationCoordinator are defined, because they are
    # accessed as soon as the routes are imported.
    import plugins.optimization_coordinator.routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
