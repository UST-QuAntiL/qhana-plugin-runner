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

import json
import jsonschema
import urllib
from functools import reduce
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Union, Any

from flask import Flask
from flask.blueprints import Blueprint
from flask.helpers import url_for
from werkzeug.utils import cached_property

from qhana_plugin_runner.util.plugins import QHAnaPluginBase
from qhana_plugin_runner.api.util import SecurityBlueprint


def template_identifier(name) -> str:
    return name.lower()


FilterExpr = Union[str, Dict[str, Any]]
# FilterExpr = Union[str, Dict[str, List[FilterExpr]], Dict[str, FilterExpr]] not yet supported by mypy
# https://github.com/python/mypy/issues/731


class QHanaTemplateCategory:
    name: ClassVar[str]
    description: ClassVar[str]
    plugin_filter: ClassVar[FilterExpr]

    def __init__(self, name: str, description: str, plugin_filter: FilterExpr) -> None:
        self.name = name
        self.description = description
        self.plugin_filter = plugin_filter

    @classmethod
    def from_dict(cls, category_dict):
        return cls(
            name=category_dict["name"],
            description=category_dict.get("description", ""),
            plugin_filter=category_dict["tags"],
        )

class QHanaTemplate:
    name: ClassVar[str]
    description: ClassVar[str]
    categories: ClassVar[List[QHanaTemplateCategory]]

    __app__: Optional[Flask] = None
    __templates__: Dict[str, "QHanaTemplate"] = {}

    def __init__(
        self,
        name: str,
        description: str,
        categories: List[QHanaTemplateCategory],
        app: Optional[Flask],
    ) -> None:
        super().__init__()
        self.app: Optional[Flask] = None
        if app:
            self.init_app(app)
        self.name = name
        self.description = description
        self.categories = categories

    def init_app(self, app: Flask) -> None:
        self.app = app

    @cached_property
    def identifier(self) -> str:
        """An url safe identifier based on name and version of the template."""
        return template_identifier(self.name)

    @staticmethod
    def get_templates() -> Dict[str, "QHanaTemplate"]:
        return QHanaTemplate.__templates__

    @classmethod
    def from_dict(cls, template_dict, app):
        categories = []

        for category_dict in template_dict["categories"]:
            category = QHanaTemplateCategory.from_dict(category_dict)
            categories.append(category)

        return cls(
            template_dict["name"], template_dict.get("description", ""), categories, app
        )

def _load_templates_from_folder(app: Flask, folder: Union[str, Path]) -> None:
    """Load all templates from a folder path.

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
        app.logger.info(
            f"Trying to load templates from '{folder}' but the folder does not exist."
        )
        return

    if not folder.is_dir():
        app.logger.warning(
            f"Trying to load templates from '{folder}' but the path is not a directory."
        )
        return

    # TODO: support multiple template folders? as implemented needs a schema file in every templates folder
    template_schema_file_name = "template_schema.json"
    try:
        template_schema_file_path = folder.joinpath(template_schema_file_name)
        with open(
            template_schema_file_path, "r", encoding="utf-8"
        ) as template_schema_file:
            template_schema = json.load(template_schema_file)
    except FileNotFoundError:
        app.logger.error(f"{template_schema_file_path} not found")
        return
    except json.decoder.JSONDecodeError:
        app.logger.error(
            f"{template_schema_file_path} template_schema.json has incorrect format"
        )
        return

    for child in folder.iterdir():
        if child.suffixes != [".json"] or child.name == "template_schema.json":
            continue

        try:
            with open(child, "r") as f:
                template = json.load(f)
                jsonschema.validate(template, template_schema)
        except json.decoder.JSONDecodeError:
            app.logger.error(f"{child} has incorrect format")
            return
        except jsonschema.exceptions.ValidationError:
            app.logger.error(
                f"{child} does not fit the schema of template configuration files defined in {template_schema_file_path}"
            )

        QHanaTemplate.__templates__[
            template_identifier(template["name"])
        ] = QHanaTemplate.from_dict(template, app)


def register_templates(app: Flask) -> None:
    """Load and register QHAna templates in the locations specified by the app config.

    Args:
        app (Flask): the app instance to register the templates with
    """
    from qhana_plugin_runner.api import ROOT_API

    QHanaTemplate.__app__ = app
    template_folders = app.config.get("TEMPLATE_FOLDERS", ["templates"])
    for folder in template_folders:
        _load_templates_from_folder(app, folder)

    url_prefix: str = app.config.get("OPENAPI_URL_PREFIX", "").rstrip("/")

    # register API blueprints (only do this after the API is registered with flask!)
    for template in QHanaTemplate.get_templates().values():
        template_blueprint = SecurityBlueprint(
            f"template-{template.identifier}-api",
            template.name,
            description=f"Api to {template.name} template.",
            url_prefix=f"{url_prefix}/templates/{template.identifier}/",
        )
        ROOT_API.register_blueprint(
            template_blueprint,
            url_prefix=f"{url_prefix}/templates/{template.identifier}/",
        )