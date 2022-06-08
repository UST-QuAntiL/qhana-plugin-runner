# Copyright 2022 QHAna plugin runner contributors.
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

"""Module containing the endpoints related to templates."""

from dataclasses import dataclass
from http import HTTPStatus
from typing import List, Dict
from unicodedata import name

import marshmallow as ma
from flask.helpers import url_for
from flask.views import MethodView
from flask_smorest import abort
from werkzeug.utils import redirect
from flask import Flask

from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint
from qhana_plugin_runner.util.templates import QHanaTemplate

TEMPLATES_API = SmorestBlueprint(
    "templates-api",
    __name__,
    description="Api to request a list of loaded templates.",
    url_prefix="/templates",
)


@dataclass()
class CategoryData:
    name: str
    description: str
    plugin_endpoints: List[str]


@dataclass()
class CategoryCollentionData:
    categories: List[CategoryData]


class CategoryDataSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    description = ma.fields.String(required=True, allow_none=False, dump_only=True)
    plugin_endpoints = ma.fields.List(
        ma.fields.String(), required=True, allow_none=False, dump_only=True
    )


class CategoryCollentionSchema(MaBaseSchema):
    categories = ma.fields.List(
        ma.fields.Nested(CategoryDataSchema()),
        required=True,
        allow_none=False,
        dump_only=True,
    )


@dataclass()
class TemplateData:
    name: str
    description: str
    identifier: str
    api_root: str


@dataclass()
class TemplateCollectionData:
    templates: List[TemplateData]


class TemplateDataSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    description = ma.fields.String(required=True, allow_none=False, dump_only=True)
    identifier = ma.fields.String(required=True, allow_none=False, dump_only=True)


class TemplateCollectionSchema(MaBaseSchema):
    templates = ma.fields.List(ma.fields.Nested(TemplateDataSchema()))


@TEMPLATES_API.route("/")
class TemplatesView(MethodView):
    """Templates collection resource."""

    @TEMPLATES_API.response(HTTPStatus.OK, TemplateCollectionSchema())
    def get(self):
        """Get all loaded templates."""

        return TemplateCollectionData(
            templates=sorted(
                [
                    TemplateData(
                        name=t.name,
                        description=t.description,
                        identifier=t.identifier,
                        api_root=url_for(
                            "templates-api.TemplateView",
                            template_id=t.identifier,
                            _external=True,
                        ),
                    )
                    for t in QHanaTemplate.get_templates().values()
                ],
                key=lambda x: x.identifier,
            )
        )


@TEMPLATES_API.route("/<string:template_id>/")
class TemplateView(MethodView):
    """Generic fallback templates view."""

    @TEMPLATES_API.response(HTTPStatus.OK, CategoryCollentionSchema())
    def get(self, template_id: str):
        """Get all loaded templates."""

        if template_id not in QHanaTemplate.get_templates():
            abort(
                HTTPStatus.NOT_FOUND,
                message=f"No template with identifier '{template_id}' exists.",
            )
        template = QHanaTemplate.get_templates()[template_id]
        return CategoryCollentionData(
            categories=[
                CategoryData(
                    name=category.name,
                    description=category.description,
                    plugin_endpoints=[
                        url_for(
                            "plugins-api.PluginView", plugin=p.identifier, _external=True
                        )
                        for p in category.plugins
                    ],
                )
                for category in template.categories
            ]
        )
