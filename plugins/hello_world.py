from typing import Optional

from flask.app import Flask

from qhana_plugin_runner.util.plugins import QHAnaPluginBase


class HelloWorld(QHAnaPluginBase):

    name = "hello-world"
    version = "0.1.0"

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)
        print("\nInitialized hello world plugin.\n")
