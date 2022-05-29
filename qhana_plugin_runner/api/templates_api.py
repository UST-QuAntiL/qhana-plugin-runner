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

"""Module containing the endpoints related to templates."""

from dataclasses import dataclass
from http import HTTPStatus
from typing import List, Optional

import marshmallow as ma
from flask.helpers import url_for
from flask.views import MethodView
from flask_smorest import abort
from werkzeug.utils import redirect
from flask import Flask

from qhana_plugin_runner.api.util import MaBaseSchema
from qhana_plugin_runner.api.util import SecurityBlueprint as SmorestBlueprint

TEMPLATES_API = SmorestBlueprint(
    "templates-api",
    __name__,
    description="Api to request a list of loaded templates.",
    url_prefix="/templates",
)


@dataclass()
class TemplateItem:
    name: str
    description: str
    tags: List[str]


@dataclass()
class TemplateData:
    title: str
    description: str
    categories: List[TemplateItem]


@dataclass()
class TemplateCollectionData:
    templates: List[TemplateData]


class TemplateItemSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    description = ma.fields.String(required=True, allow_none=False, dump_only=True)
    tags = ma.fields.List(ma.fields.String())


class TemplateSchema(MaBaseSchema):
    title = ma.fields.String(required=True, allow_none=False, dump_only=True)
    description = ma.fields.String(required=True, allow_none=False, dump_only=True)
    categories = ma.fields.List(ma.fields.Nested(TemplateItemSchema()))


class TemplateCollectionSchema(MaBaseSchema):
    templates = ma.fields.List(ma.fields.Nested(TemplateSchema()))


@TEMPLATES_API.route("/")
class TemplatesView(MethodView):
    """Templates collection resource."""

    @TEMPLATES_API.response(HTTPStatus.OK, TemplateCollectionSchema())
    def get(self):
        """Get all loaded templates."""
        template_data = [
            {
                "title": "MUSE",
                "description": "Template for MUSE workflow",
                "categories": [
                    {
                        "name": "Load data",
                        "description": "Plugin for loading costume data",
                        "tags": [
                            "data:loading"
                        ]
                    },
                    {
                        "name": "Data Preperation",
                        "description": "Plugins for data Perperation",
                        "tags": [
                            "similarity-cache-generation",
                            "similarity-calculation",
                            "attribute-similarity-calculation",
                            "sim-to-dist",
                            "aggregator",
                            "dist-to-points"
                        ]
                    },
                    {
                        "name": "Quantum Part",
                        "description": "Plugin for Quantum Algorithm",
                        "tags": [
                            "points-to-clusters"
                        ]
                    },
                    {
                        "name": "Visualization",
                        "description": "Plugin for visualization",
                        "tags": [
                            "visualization"
                        ]
                    }
                ]
            }
        ]

        return TemplateCollectionData(
            templates=[
                TemplateData(
                    title=t['title'],
                    description=t['description'],
                    categories=[
                        TemplateItem(
                            name=c['name'],
                            description=c['description'],
                            tags=c['tags']
                        ) for c in t['categories']
                    ]
                )
                for t in template_data
            ]
        )
