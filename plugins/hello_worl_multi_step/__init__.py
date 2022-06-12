from typing import Optional

from flask import Flask

from qhana_plugin_runner.api.util import SecurityBlueprint
from qhana_plugin_runner.util.plugins import plugin_identifier, QHAnaPluginBase

_plugin_name = "hello-world-multi-step"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


HELLO_MULTI_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="Demo plugin API.",
    template_folder="hello_world_templates",
)


class HelloWorldMultiStep(QHAnaPluginBase):

    name = _plugin_name
    version = __version__
    tags = []

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return HELLO_MULTI_BLP


try:
    # It is important to import the routes **after** COSTUME_LOADER_BLP and CostumeLoader are defined, because they are
    # accessed as soon as the routes are imported.
    from . import routes
except ImportError:
    # When running `poetry run flask install`, importing the routes will fail, because the dependencies are not
    # installed yet.
    pass
