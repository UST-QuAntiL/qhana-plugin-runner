from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

from flask import render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)

from .blueprint import REST_CONN_BLP
from .schemas import WelcomeParametersSchema
from ..database import get_wip_connectors, start_new_connector
from ..plugin import RESTConnector


@REST_CONN_BLP.route("/")
class PluginRootView(MethodView):
    """Plugin for managing plugins connecting to REST APIs."""

    @REST_CONN_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata"""

        return PluginMetadata(
            title="REST API Connector",
            description="Integrate REST APIs as plugins.",
            name=RESTConnector.instance.name,
            version=RESTConnector.instance.version,
            type=PluginType.interaction,
            entry_point=EntryPoint(
                href=url_for(
                    f"{REST_CONN_BLP.name}.{WelcomeView.__name__}", _external=True
                ),
                ui_href=url_for(
                    f"{REST_CONN_BLP.name}.{WelcomeFrontend.__name__}", _external=True
                ),
                data_input=[],
                data_output=[],
            ),
            tags=RESTConnector.instance.tags,
        )


@REST_CONN_BLP.route("/wip-connectors-ui/")
class WelcomeFrontend(MethodView):
    """Micro frontend for the welcome page of the REST connector plugin."""

    @REST_CONN_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the welcome page of the REST connector plugin.",
    )
    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""

        return self.render({}, {}, True)

    @REST_CONN_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the welcome page of the REST connector plugin.",
    )
    @REST_CONN_BLP.arguments(
        WelcomeParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with pre-rendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = RESTConnector.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        services = []  # FIXME
        ongoing = [
            (
                v,
                url_for(
                    f"{REST_CONN_BLP.name}.WipConnectorUiView",
                    connector_id=k,
                    _external=True,
                ),
                url_for(
                    f"{REST_CONN_BLP.name}.WipConnectorView",
                    connector_id=k,
                    _external=True,
                ),
            )
            for k, v in get_wip_connectors().items()
        ]

        ongoing.sort()

        schema = WelcomeParametersSchema()
        return Response(
            render_template(
                "rest_connector_welcome.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{REST_CONN_BLP.name}.{WelcomeView.__name__}"),
                services=services,
                ongoing=ongoing,
            )
        )


@REST_CONN_BLP.route("/wip-connectors/")
class WelcomeView(MethodView):
    """TODO."""

    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        services = []  # FIXME
        ongoing = [
            {
                "name": v,
                "connector_id": k,
                "href": url_for(
                    f"{REST_CONN_BLP.name}.WipConnectorView",
                    connector_id=k,
                    _external=True,
                ),
                "ui_href": url_for(
                    f"{REST_CONN_BLP.name}.WipConnectorUiView",
                    connector_id=k,
                    _external=True,
                ),
            }
            for k, v in get_wip_connectors().items()
        ]
        return {
            "services": services,
            "ongoing": ongoing,
        }

    @REST_CONN_BLP.arguments(WelcomeParametersSchema(), location="form")
    def post(self, data):
        name = data["api_name"]
        connector_id = start_new_connector(name, commit=True)
        return {
            "name": name,
            "connector_id": connector_id,
            "href": url_for(
                f"{REST_CONN_BLP.name}.WipConnectorView",
                connector_id=connector_id,
                _external=True,
            ),
            "ui_href": url_for(
                f"{REST_CONN_BLP.name}.WipConnectorUiView",
                connector_id=connector_id,
                _external=True,
            ),
        }
