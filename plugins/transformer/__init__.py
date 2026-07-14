from typing import Optional
from flask.app import Flask
from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

_plugin_name = "element_sim-to-element_dist-transformers"
__version__ = "v0.0.1"
_identifier = plugin_identifier(_plugin_name, __version__)


ELEMENT_TRANSFORMERS_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Element similarities to element distances transformers plugin API.",
)


class ElementTransformers(QHAnaPluginBase):
    name = _plugin_name
    version = __version__
    description = "Transforms element similarities to element distances."
    tags = ["preprocessing", "similarity-calculation", "distance-calculation"]

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return ELEMENT_TRANSFORMERS_BLP

    def get_requirements(self) -> str:
        return ""


try:
    from . import routes  # noqa: F401,E402
except ImportError:
    # When running `poetry run flask install`,
    # importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
