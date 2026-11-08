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

# originally from <https://github.com/buehlefs/flask-template/>

"""Module containing all API related code of the project."""

from http import HTTPStatus
from typing import Dict

import marshmallow as ma
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec.ext.marshmallow.common import resolve_schema_instance
from flask import Flask
from flask.views import MethodView
from flask_smorest import Api
from flask_smorest import Blueprint as SmorestBlueprint

from .extra_fields import EnumField
from .files_api import FILES_API
from .jwt_helper import SECURITY_SCHEMES
from .plugins_api import PLUGINS_API
from .tasks_api import TASKS_API
from .util import MaBaseSchema


def _schema_name_resolver(schema) -> str:
    """Build a unique OpenAPI component name for a marshmallow schema.

    The apispec default resolver uses the bare class name, which collides
    both across plugins (``InputParametersSchema`` exists many times) and
    within one class registered with different modifiers. Qualifying the
    name with the module and the modifiers keeps component names unique.
    """
    instance = resolve_schema_instance(schema)
    schema_cls = type(instance)
    name = schema_cls.__name__.removesuffix("Schema")
    parts = [f"{schema_cls.__module__}.{name}"]
    if instance.partial:
        parts.append(
            "Partial"
            if instance.partial is True
            else "Partial-" + "-".join(sorted(instance.partial))
        )
    if instance.only:
        parts.append("Only-" + "-".join(sorted(instance.only)))
    if instance.exclude:
        parts.append("Exclude-" + "-".join(sorted(instance.exclude)))
    if instance.load_only:
        parts.append("LoadOnly-" + "-".join(sorted(instance.load_only)))
    if instance.dump_only:
        parts.append("DumpOnly-" + "-".join(sorted(instance.dump_only)))
    return "-".join(parts)


"""A single API instance. All api versions should be blueprints."""
ROOT_API = Api(
    spec_kwargs={
        "title": "QHAna plugin runner api.",
        "version": "v1",
        "marshmallow_plugin": MarshmallowPlugin(
            schema_name_resolver=_schema_name_resolver
        ),
    }
)


ROOT_API.register_field(EnumField, "string", None)


class VersionsRootSchema(MaBaseSchema):
    title = ma.fields.String(required=True, allow_none=False, dump_only=True)


ROOT_ENDPOINT = SmorestBlueprint(
    "api-root",
    __name__,
    url_prefix="/api",
    description="The API endpoint pointing towards all api versions.",
)


@ROOT_ENDPOINT.route("/")
class RootView(MethodView):
    @ROOT_ENDPOINT.response(HTTPStatus.OK, VersionsRootSchema())
    def get(self) -> Dict[str, str]:
        """Get the Root API information containing the links other endpoints of this api."""
        return {
            "title": ROOT_API.spec.title,
        }


def register_root_api(app: Flask):
    """Register the API with the flask app."""
    ROOT_API.init_app(app)

    # register security schemes in doc
    for name, scheme in SECURITY_SCHEMES.items():
        ROOT_API.spec.components.security_scheme(name, scheme)

    url_prefix: str = app.config.get("OPENAPI_URL_PREFIX", "").rstrip("/")

    # register API blueprints (only do this after the API is registered with flask!)
    ROOT_API.register_blueprint(ROOT_ENDPOINT, url_prefix=url_prefix)
    ROOT_API.register_blueprint(PLUGINS_API, url_prefix=f"{url_prefix}/plugins")
    ROOT_API.register_blueprint(TASKS_API, url_prefix=f"{url_prefix}/tasks")
    ROOT_API.register_blueprint(FILES_API, url_prefix=f"{url_prefix}/files")
