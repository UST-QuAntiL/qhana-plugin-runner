# Copyright 2021 QHAna plugin runner contributors.
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

import sys
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union

from flask import Flask
from werkzeug.urls import url_quote
from werkzeug.utils import cached_property


class QHAnaPluginBase():

    name: ClassVar[str]
    version: ClassVar[str]

    __app__: Optional[Flask] = None
    __plugins__: Dict[str, "QHAnaPluginBase"] = {}

    def __init_subclass__(cls) -> None:
        try:
            plugin = cls(app=QHAnaPluginBase.__app__)
            if not plugin.name:
                raise ValueError("A plugin must specify a URL-safe! name.")
            if not plugin.version:
                raise ValueError("A plugin must specify a version.")
            QHAnaPluginBase.__plugins__[plugin.identifier] = plugin  # TODO better vetting/error checking
        except Exception:
            if QHAnaPluginBase.__app__:
                QHAnaPluginBase.__app__.logger.info("Could not load the plugin class {}!", cls)
            else:
                print(f"Could not load plugin class {cls}!")

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__()
        self.app: Optional[Flask] = None
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        self.app = app

    def get_plugins() -> Dict[str, "QHAnaPluginBase"]:
        return QHAnaPluginBase.__plugins__

    @cached_property
    def identifier(self) -> str:
        return url_quote(f"{self.name}_{self.version}")


def _load_plugins_from_folder(folder: Union[str, Path]):
    # TODO implement plugin loading from a folder containing plugins
    raise NotImplementedError()


def register_plugins(app: Flask):
    QHAnaPluginBase.__app__ = app
    plugin_folders = app.config.get("PLUGIN_FOLDERS", [])
    for folder in plugin_folders:
        _load_plugins_from_folder(folder)

    # manually load plugin until plugin discovery is implemented
    sys.path.append(str(Path('./plugins').absolute()))
    import hello_world

