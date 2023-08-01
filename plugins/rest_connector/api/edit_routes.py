from collections import ChainMap
from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

from flask import render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from marshmallow import EXCLUDE, RAISE

from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.virtual_plugins import PluginState, VirtualPlugin

from .blueprint import REST_CONN_BLP
from .schemas import (
    ConnectorKey,
    ConnectorSchema,
    ConnectorUpdateSchema,
    ConnectorVariableSchema,
    RequestFileDescriptorSchema,
    ResponseOutputSchema,
)
from ..database import get_wip_connectors, save_wip_connectors, start_new_connector
from ..plugin import RESTConnector


@REST_CONN_BLP.route("/wip-connectors-ui/<string:connector_id>/")
class WipConnectorUiView(MethodView):
    """UI for editing work in progress REST connector definitions."""

    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def get(self, connector_id: str):
        wip_connectors = get_wip_connectors()

        try:
            name = wip_connectors[connector_id]
        except KeyError:
            abort(
                HTTPStatus.NOT_FOUND,
                message=f"Found no connector with id '{connector_id}'.",
            )

        plugin = RESTConnector.instance
        connector = PluginState.get_value(plugin.identifier, connector_id, default={})
        assert isinstance(connector, dict), "Type assertion"
        return Response(
            render_template(
                "rest_connector_edit.html",
                name=name,
                connector=ConnectorSchema().dump(connector),
                process=url_for(
                    f"{REST_CONN_BLP.name}.{WipConnectorView.__name__}",
                    connector_id=connector_id,
                ),
            )
        )


@REST_CONN_BLP.route("/wip-connectors/<string:connector_id>/")
class WipConnectorView(MethodView):
    """API for editing work in progress REST connector definitions."""

    @REST_CONN_BLP.response(HTTPStatus.OK, ConnectorSchema())
    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def get(self, connector_id: str):
        wip_connectors = get_wip_connectors()

        try:
            name = wip_connectors[connector_id]
        except KeyError:
            abort(
                HTTPStatus.NOT_FOUND,
                message=f"Found no connector with id '{connector_id}'.",
            )

        plugin = RESTConnector.instance
        connector = PluginState.get_value(plugin.identifier, connector_id, default={})
        assert isinstance(connector, dict), "Type assertion"
        return ChainMap(connector, {"name": name})

    @REST_CONN_BLP.response(HTTPStatus.OK, ConnectorSchema())
    @REST_CONN_BLP.arguments(ConnectorUpdateSchema(), location="json")
    def post(self, data, connector_id: str):
        wip_connectors = get_wip_connectors()

        try:
            name = wip_connectors[connector_id]
        except KeyError:
            abort(
                HTTPStatus.NOT_FOUND,
                message=f"Found no connector with id '{connector_id}'.",
            )

        plugin = RESTConnector.instance
        connector = PluginState.get_value(plugin.identifier, connector_id, default={})
        assert isinstance(connector, dict), "Type assertion"

        update_key: ConnectorKey = data["key"]
        update_value = data["value"]

        assert isinstance(update_value, str)

        if connector.get("is_deployed", False):
            if update_key != ConnectorKey.UNDEPLOY:
                abort(
                    HTTPStatus.BAD_REQUEST,
                    message="Cannot edit a currently deployed REST Connector. Please undeploy the connector first.",
                )

        if update_key == ConnectorKey.NAME:
            wip_connectors[connector_id] = update_value
            save_wip_connectors(wip_connectors, commit=True)
        elif update_key == ConnectorKey.BASE_URL:
            connector = self.update_base_url(connector, update_value)
        elif update_key == ConnectorKey.OPENAPI_SPEC:
            connector = self.update_openapi_spec(connector, update_value)
        elif update_key == ConnectorKey.ENDPOINT_URL:
            connector = self.update_endpoint_url(connector, update_value)
        elif update_key == ConnectorKey.VARIABLES:
            connector = self.update_variables(connector, update_value)
        elif update_key == ConnectorKey.REQUEST_HEADERS:
            connector = self.update_request_headers(connector, update_value)
        elif update_key == ConnectorKey.REQUEST_BODY:
            connector = self.update_request_body(connector, update_value)
        elif update_key == ConnectorKey.REQUEST_FILES:
            connector = self.update_request_files(connector, update_value)
        elif update_key == ConnectorKey.RESPONSE_HANDLING:
            connector = self.update_response_handling(connector, update_value)
        elif update_key == ConnectorKey.RESPONSE_MAPPING:
            connector = self.update_response_mapping(connector, update_value)
        elif update_key == ConnectorKey.DEPLOY:
            connector = self.deploy_as_plugin(
                connector,
                connector_id=connector_id,
                parent_id=plugin.identifier,
            )
        elif update_key == ConnectorKey.UNDEPLOY:
            connector = self.undeploy_plugin(
                connector, connector_id=connector_id, parent_id=plugin.identifier
            )
        elif update_key == ConnectorKey.CANCEL:
            # TODO cancel background task currently locking the connector
            abort(HTTPStatus.NOT_IMPLEMENTED, "TODO: Command not implemented yet.")

        if update_key != ConnectorKey.NAME:
            connector["next_step"] = data.get("next_step", "")
            PluginState.set_value(plugin.identifier, connector_id, connector, commit=True)

        return ChainMap(connector, {"name": name})

    def update_base_url(self, connector: dict, new_base_url: str) -> dict:
        connector["base_url"] = new_base_url
        return connector

    def update_openapi_spec(self, connector: dict, api_spec_url: str) -> dict:
        old_url = connector.get("openapi_spec_url", "")
        if old_url:
            pass  # TODO remove old URL if it is hosted on the plugin runner
        if api_spec_url == "file":
            api_spec_file = next(request.files.values())
            # TODO handle this case
            raise NotImplementedError()
        connector["openapi_spec_url"] = api_spec_url
        return connector

    def update_endpoint_url(self, connector: dict, new_endpoint_url: str) -> dict:
        connector["endpoint_url"] = new_endpoint_url
        # TODO: discover headers/body from openapi spec?
        return connector

    def update_variables(self, connector: dict, new_variables: str) -> dict:
        print(new_variables)
        parsed_variables = ConnectorVariableSchema(many=True).loads(new_variables)
        connector["variables"] = parsed_variables
        return connector

    def update_request_headers(self, connector: dict, new_headers: str) -> dict:
        # TODO validate headers?
        connector["request_headers"] = new_headers
        return connector

    def update_request_body(self, connector: dict, new_body: str) -> dict:
        connector["request_body"] = new_body
        return connector

    def update_request_files(self, connector: dict, new_files: str) -> dict:
        # TODO remove old file URLs if they are hosted on the plugin runner
        parsed_files = RequestFileDescriptorSchema(many=True).loads(new_files)
        connector["request_files"] = parsed_files
        return connector

    def update_response_handling(
        self, connector: dict, new_response_handling: str
    ) -> dict:
        connector["response_handling"] = new_response_handling
        return connector

    def update_response_mapping(self, connector: dict, new_response_mapping: str) -> dict:
        parsed_response_mapping = ResponseOutputSchema(many=True).loads(
            new_response_mapping
        )
        connector["response_mapping"] = parsed_response_mapping
        return connector

    def deploy_as_plugin(
        self, connector: dict, connector_id: str, parent_id: str
    ) -> dict:
        if connector.get("is_deployed", False):
            return connector

        version = connector.get("version", 0) + 1
        connector["version"] = version

        connector["is_deployed"] = True
        # TODO move connector from WIP connectors list to published connectors list

        plugin_url = url_for(
            f"{REST_CONN_BLP.name}.VirtualPluginView",
            connector_id=connector_id,
            _external=True,
        )

        if VirtualPlugin.exists([VirtualPlugin.href == plugin_url]):
            return connector

        plugin = VirtualPlugin(
            parent_id=parent_id,
            version=str(version),
            name=connector_id,
            description=connector.get("description", ""),
            tags="\n".join(["rest-connector", "TODO"]),  # TODO allow users to edit tags
            href=plugin_url,
        )

        DB.session.add(plugin)
        DB.session.commit()
        return connector

    def undeploy_plugin(self, connector: dict, connector_id: str, parent_id: str) -> dict:
        plugin_url = url_for(
            f"{REST_CONN_BLP.name}.VirtualPluginView",
            connector_id=connector_id,
            _external=True,
        )

        if VirtualPlugin.exists([VirtualPlugin.href == plugin_url]):
            VirtualPlugin.delete_by_href(plugin_url, parent_id=parent_id)
            DB.session.commit()

        del connector["is_deployed"]
        # TODO move connector from published connectors list to WIP connectors list
        return connector
