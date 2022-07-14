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

from dataclasses import dataclass
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


def create_identifier(name) -> str:
    return name.lower().replace(" ", "-")


FilterExpr = Union[str, Dict[str, Any]]
# FilterExpr = Union[str, Dict[str, List[FilterExpr]], Dict[str, FilterExpr]] not yet supported by mypy
# https://github.com/python/mypy/issues/731


@dataclass()
class QHanaTemplateCategory:
    name: str
    description: str
    plugin_filter: FilterExpr

    @cached_property
    def identifier(self) -> str:
        """An url safe identifier based on name of the category."""
        return create_identifier(self.name)

    @classmethod
    def from_dict(cls, category_dict):
        return cls(
            name=category_dict["name"],
            description=category_dict.get("description", ""),
            plugin_filter=category_dict["filter"],
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
        """An url safe identifier based on name of the template."""
        return create_identifier(self.name)

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


def _load_template_schema() -> Dict[str, Any]:
    template_schema_file_path = Path("templates/schema/template_schema.json")
    template_schema_file_path = template_schema_file_path.resolve()

    try:
        with open(
            template_schema_file_path, "r", encoding="utf-8"
        ) as template_schema_file:
            return json.load(template_schema_file)

    except FileNotFoundError:
        app.logger.error(f"{template_schema_file_path} not found")
    except json.decoder.JSONDecodeError:
        app.logger.error(
            f"{template_schema_file_path} template_schema.json has incorrect format"
        )


def _load_templates_from_folder(
    app: Flask, folder: Union[str, Path], template_schema: Dict[str, Any]
) -> None:
    """Load all templates from a folder path.

    Every .json file in the folder is considered a template and will be loaded.

    Args:
        app (Flask): the app instance (used for logging only)
        folder (Union[str, Path]): the folder path to scan for templates
        template_schema (Dict[str, Any]): the schema the templates have to conform to
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
            create_identifier(template["name"])
        ] = QHanaTemplate.from_dict(template, app)


def register_templates(app: Flask) -> None:
    """Load and register QHAna templates in the locations specified by the app config.

    Args:
        app (Flask): the app instance to register the templates with
    """
    from qhana_plugin_runner.api import ROOT_API

    QHanaTemplate.__app__ = app

    template_schema = _load_template_schema()

    template_folders = app.config.get("TEMPLATE_FOLDERS", ["templates"])
    for folder in template_folders:
        _load_templates_from_folder(app, folder, template_schema)
