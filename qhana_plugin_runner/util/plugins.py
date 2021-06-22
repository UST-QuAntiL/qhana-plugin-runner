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
from importlib import import_module
from pathlib import Path
from typing import ClassVar, Dict, Optional, Union

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
        """An url safe identifier based on name and version of the plugin."""
        return url_quote(f"{self.name}_{self.version}")


def _append_source_path(app: Flask, source_path: Union[str, Path]):
    """Add a new source path containing python modules to the system path avoiding duplicates.

    Args:
        app (Flask): the app instance (used for logging only)
        source_path (Union[str, Path]): the path to add
    """
    source_path = str(source_path)
    if source_path in sys.path:
        return
    app.logger.debug("Adding new source path to the python path: '{}'", source_path)
    sys.path.append(source_path)


def _try_load_plugin_file(app: Flask, plugin_file: Path):
    """Try to load a single file python module.

    If the file has an unknown file ending then importing the file will not be attempted.

    Args:
        app (Flask): the app instance (used for logging only)
        plugin_file (Path): the path to the python module file (not the directory containing the file!)
    """
    if plugin_file.suffixes == [".py"]:
        try:
            import_module(plugin_file.stem)
        except ImportError:
            app.logger.warning("Failed to import '{}' at location '{}'.", plugin_file.name, plugin_file.parent)


def _try_load_plugin_package(app: Flask, plugin_package: Path):
    """Try to load a plugin package (a folder containing an ``__init__.py`` file)

    If ``plugin_package/__init__.py`` does not exist this method does nothing.

    Args:
        app (Flask): the app instance (used for logging only)
        plugin_package (Path): the path to the python package (not the directory containing the package!)
    """
    if not (plugin_package / Path("__init__.py")).exists():
        app.logger.debug("Tried to import a normal folder '{}' as a python package, skipping.", plugin_package)
        return
    try:
        import_module(plugin_package.name)
    except ImportError:
        app.logger.warning("Failed to import '{}' at location '{}'.", plugin_package.name, plugin_package.parent)


def _load_plugins_from_folder(app: Flask, folder: Union[str, Path]):
    """Load all plugins from a folder path.

    Every importable python module in the folder is considered a plugin and 
    will be automatically imported. The parent path will be added to ``sys.path``
    if not already added.

    If the folder contains a ``__init__.py`` file itself, then the folder is 
    assumed to be a python package and only that python package is imported.

    Args:
        app (Flask): the app instance (used for logging only)
        folder (Union[str, Path]): the folder path to scan for plugins
    """
    if isinstance(folder, str):
        folder = Path(folder)
    
    folder = folder.resolve()

    if not folder.exists():
        app.logger.info("Trying to load plugins from '{}' but the folder does not exist.", folder)
    if not folder.is_dir():
        app.logger.warning("Trying to load plugins from '{}' but the path is not a directory.", folder)

    if (folder / Path("__init__.py")).exists():
        _append_source_path(app, folder.parent)
        _try_load_plugin_package(app, folder)
        return

    _append_source_path(app, folder)

    for child in folder.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_file():
            _try_load_plugin_file(app, child)
        if child.is_dir():
            _try_load_plugin_package(app, child)


def register_plugins(app: Flask):
    """Load and register QHAna plugins in the locations specified by the app config.

    Args:
        app (Flask): the app instance to register the plugins with
    """
    QHAnaPluginBase.__app__ = app
    plugin_folders = app.config.get("PLUGIN_FOLDERS", [])
    for folder in plugin_folders:
        _load_plugins_from_folder(app, folder)

