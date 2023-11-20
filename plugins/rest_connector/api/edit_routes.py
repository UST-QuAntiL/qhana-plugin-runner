from collections import ChainMap
from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

from celery.canvas import Signature
from celery.exceptions import TimeoutError
from celery.result import AsyncResult
from flask import render_template
from flask.globals import current_app, request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from marshmallow import EXCLUDE, RAISE

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.virtual_plugins import (
    VIRTUAL_PLUGIN_CREATED,
    VIRTUAL_PLUGIN_REMOVED,
    PluginState,
    VirtualPlugin,
)

from .blueprint import REST_CONN_BLP
from .prefill_tasks import prefill_values, unlock_connector
from .schemas import (
    ConnectorKey,
    ConnectorSchema,
    ConnectorUpdateSchema,
    ConnectorVariableSchema,
    RequestFileDescriptorSchema,
    ResponseOutputSchema,
)
from ..database import (
    deploy_connector,
    get_deployed_connector,
    get_wip_connectors,
    remove_wip_connector,
    save_wip_connectors,
    undeploy_connector,
)
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
            name = get_deployed_connector(connector_id)
            if name is None:
                abort(
                    HTTPStatus.NOT_FOUND,
                    message=f"Found no connector with id '{connector_id}'.",
                )

        plugin = RESTConnector.instance
        connector = PluginState.get_value(plugin.identifier, connector_id, default={})
        connector_extra = PluginState.get_value(
            plugin.identifier, f"{connector_id}__extra", default={}
        )
        assert isinstance(connector, dict), "Type assertion"
        assert isinstance(connector_extra, dict), "Type assertion"
        return Response(
            render_template(
                "rest_connector_edit.html",
                name=name,
                connector=ConnectorSchema().dump(connector),
                extra=connector_extra,
                has_assistant=True,  # TODO check if ai assistant is actually available
                http_methods=["GET", "PUT", "POST", "DELETE", "PATCH"],
                process=url_for(
                    f"{REST_CONN_BLP.name}.{WipConnectorView.__name__}",
                    connector_id=connector_id,
                ),
                back=url_for(f"{REST_CONN_BLP.name}.WelcomeFrontend"),
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
            name = get_deployed_connector(connector_id)
            if name is None:
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
            name = get_deployed_connector(connector_id)
            if name is None:
                abort(
                    HTTPStatus.NOT_FOUND,
                    message=f"Found no connector with id '{connector_id}'.",
                )

        plugin = RESTConnector.instance
        connector = PluginState.get_value(plugin.identifier, connector_id, default={})
        assert isinstance(connector, dict), "Type assertion"

        update_key: ConnectorKey = data["key"]
        update_value = data["value"]

        ai_assist = data["ai_assist"]

        assert isinstance(update_value, str)

        if connector.get("is_deployed", False):
            if update_key != ConnectorKey.UNDEPLOY:
                abort(
                    HTTPStatus.BAD_REQUEST,
                    message="Cannot edit a currently deployed REST Connector. Please undeploy the connector first.",
                )

        if not connector.get("is_loading", False):
            if ai_assist:
                # TODO update connector using AI
                pass
            else:
                connector = self.update_connector(
                    connector_id,
                    wip_connectors,
                    plugin,
                    connector,
                    update_key,
                    update_value,
                )
        else:
            if update_key != ConnectorKey.CANCEL:
                abort(
                    HTTPStatus.BAD_REQUEST,
                    message="Cannot edit a busy REST Connector. Please cancel background tasks first or wait for them to finish!",
                )

        if update_key == ConnectorKey.CANCEL:
            task_id = PluginState.get_value(
                plugin.identifier, f"{connector_id}_bg_task"
            )
            if task_id:
                result = AsyncResult(task_id, app=CELERY)
                result.revoke(terminate=True)
                PluginState.delete_value(
                    plugin.identifier, f"{connector_id}_bg_task", commit=True
                )
            connector["is_loading"] = False
            data["next_step"] = connector.get("next_step", data.get("next_step", ""))

        connector = self.start_autofill_task(
            data, connector_id, plugin, connector, update_key
        )

        return ChainMap(connector, {"name": name})

    @REST_CONN_BLP.response(HTTPStatus.NO_CONTENT)
    def delete(self, connector_id: str):
        wip_connectors = get_wip_connectors()

        try:
            name = wip_connectors[connector_id]
        except KeyError:
            name = get_deployed_connector(connector_id)
            if name is None:
                # Treat not found as already deleted
                return HTTPStatus.NO_CONTENT

        plugin = RESTConnector.instance
        connector = PluginState.get_value(plugin.identifier, connector_id, default={})
        assert isinstance(connector, dict), "Type assertion"

        if connector.get("is_deployed", False):
            abort(
                HTTPStatus.BAD_REQUEST,
                message="Cannot delete a currently deployed REST Connector. Please undeploy the connector first.",
            )

        if not connector.get("is_loading", False):
            abort(
                HTTPStatus.BAD_REQUEST,
                message="Cannot delete a currently busy REST Connector. Please cancel any background task first or wait for them to finish.",
            )

        # delete DB entries
        PluginState.delete_value(plugin.identifier, connector_id)
        PluginState.delete_value(plugin.identifier, f"{connector_id}__extra")
        remove_wip_connector(connector_id, commit=True)

    def update_connector(
        self, connector_id, wip_connectors, plugin, connector, update_key, update_value
    ):
        if update_key == ConnectorKey.NAME:  # TODO support tag updates
            wip_connectors[connector_id] = update_value
            save_wip_connectors(wip_connectors, commit=True)
        elif update_key == ConnectorKey.BASE_URL:
            connector = self.update_base_url(connector, update_value)
        elif update_key == ConnectorKey.OPENAPI_SPEC:
            connector = self.update_openapi_spec(connector, update_value)
        elif update_key == ConnectorKey.ENDPOINT_URL:
            connector = self.update_endpoint_url(connector, update_value)
        elif update_key == ConnectorKey.ENDPOINT_METHOD:
            connector = self.update_endpoint_method(connector, update_value)
        elif update_key == ConnectorKey.VARIABLES:
            connector = self.update_variables(connector, update_value)
        elif update_key == ConnectorKey.ENDPOINT_VARIABLES:
            connector = self.update_endpoint_variables(connector, update_value)
        elif update_key == ConnectorKey.ENDPOINT_QUERY_VARIABLES:
            connector = self.update_endpoint_query_variables(connector, update_value)
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

        return connector

    def start_autofill_task(self, data, connector_id, plugin, connector, update_key):
        step_has_autofill = update_key in (
            ConnectorKey.OPENAPI_SPEC,
            ConnectorKey.ENDPOINT_URL,
            ConnectorKey.ENDPOINT_METHOD,
        )
        if step_has_autofill:
            connector["is_loading"] = True

        if update_key != ConnectorKey.NAME or step_has_autofill:
            connector["next_step"] = data.get("next_step", "")
            PluginState.set_value(
                plugin.identifier, connector_id, connector, commit=True
            )

        if step_has_autofill:
            task: Signature = prefill_values.s(
                connector_id=connector_id, last_step=update_key.value
            )
            task.link_error(unlock_connector.si(connector_id=connector_id))
            result = cast(Optional[AsyncResult], task.apply_async(expires=300))
            if result:
                PluginState.set_value(
                    plugin.identifier, f"{connector_id}_bg_task", result.id, commit=True
                )
                try:
                    result.get(timeout=3, propagate=False)
                    # reload connector to get changes
                    connector = PluginState.get_value(
                        plugin.identifier, connector_id, default={}
                    )
                except TimeoutError:
                    pass
        return connector

    def update_base_url(self, connector: dict, new_base_url: str) -> dict:
        connector["base_url"] = new_base_url
        connector["finished_steps"] = [ConnectorKey.BASE_URL.value]
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

        finished = set(connector.get("finished_steps", []))
        finished.add(ConnectorKey.OPENAPI_SPEC.value)
        connector["finished_steps"] = list(
            finished.intersection(
                {ConnectorKey.BASE_URL.value, ConnectorKey.OPENAPI_SPEC.value}
            )
        )
        return connector

    def update_endpoint_url(self, connector: dict, new_endpoint_url: str) -> dict:
        connector["endpoint_url"] = new_endpoint_url

        finished = set(connector.get("finished_steps", []))
        finished.add(ConnectorKey.ENDPOINT_URL.value)
        connector["finished_steps"] = list(
            finished.intersection(
                {
                    ConnectorKey.BASE_URL.value,
                    ConnectorKey.OPENAPI_SPEC.value,
                    ConnectorKey.ENDPOINT_URL.value,
                }
            )
        )
        return connector

    def update_endpoint_method(self, connector: dict, new_endpoint_method: str) -> dict:
        connector["endpoint_method"] = new_endpoint_method

        finished = set(connector.get("finished_steps", []))
        finished.add(ConnectorKey.ENDPOINT_METHOD.value)
        connector["finished_steps"] = list(
            finished.intersection(
                {
                    ConnectorKey.BASE_URL.value,
                    ConnectorKey.OPENAPI_SPEC.value,
                    ConnectorKey.ENDPOINT_URL.value,
                    ConnectorKey.ENDPOINT_METHOD.value,
                }
            )
        )
        return connector

    def update_endpoint_variables(
        self, connector: dict, new_endpoint_variables: str
    ) -> dict:
        parsed = current_app.json.loads(new_endpoint_variables)
        if not isinstance(parsed, dict):
            raise TypeError(
                "Endpoint variables must be a JSON object with string keys and values!"
            )
        bad_keys = {str(k) for k in parsed.keys() if not isinstance(k, str)}
        bad_values = [
            f"{v} [key: {k}]" for k, v in parsed.items() if not isinstance(v, str)
        ]
        if bad_keys or bad_values:
            message = (
                "Endpoint variables must be a JSON object with string keys and values! ("
                + ", ".join(
                    (
                        "Bad Keys: " + ", ".join(bad_keys),
                        "Bad Values:" + ", ".join(bad_values),
                    )
                )
                + ")"
            )
            raise TypeError(message)

        connector["endpoint_variables"] = parsed

        finished = connector.get("finished_steps", [])
        finished.append(ConnectorKey.ENDPOINT_VARIABLES.value)
        connector["finished_steps"] = finished
        # TODO: check variables against endpoint template?
        return connector

    def update_endpoint_query_variables(
        self, connector: dict, new_query_variables: str
    ) -> dict:
        parsed = current_app.json.loads(new_query_variables)
        if not isinstance(parsed, dict):
            raise TypeError(
                "Endpoint query variables must be a JSON object with string keys and values!"
            )
        bad_keys = {str(k) for k in parsed.keys() if not isinstance(k, str)}
        bad_values = [
            f"{v} [key: {k}]" for k, v in parsed.items() if not isinstance(v, str)
        ]
        if bad_keys or bad_values:
            message = (
                "Endpoint query variables must be a JSON object with string keys and values! ("
                + ", ".join(
                    (
                        "Bad Keys: " + ", ".join(bad_keys),
                        "Bad Values:" + ", ".join(bad_values),
                    )
                )
                + ")"
            )
            raise TypeError(message)

        connector["endpoint_query_variables"] = parsed

        finished = connector.get("finished_steps", [])
        finished.append(ConnectorKey.ENDPOINT_QUERY_VARIABLES.value)
        connector["finished_steps"] = finished
        # TODO: check variables against endpoint definition in openapi spec?
        return connector

    def update_variables(self, connector: dict, new_variables: str) -> dict:
        parsed_variables = ConnectorVariableSchema(many=True).loads(new_variables)
        connector["variables"] = parsed_variables

        finished = connector.get("finished_steps", [])
        finished.append(ConnectorKey.VARIABLES.value)
        connector["finished_steps"] = finished
        return connector

    def update_request_headers(self, connector: dict, new_headers: str) -> dict:
        # TODO validate headers?
        connector["request_headers"] = new_headers

        finished = connector.get("finished_steps", [])
        finished.append(ConnectorKey.REQUEST_HEADERS.value)
        connector["finished_steps"] = finished
        return connector

    def update_request_body(self, connector: dict, new_body: str) -> dict:
        connector["request_body"] = new_body

        finished = connector.get("finished_steps", [])
        finished.append(ConnectorKey.REQUEST_BODY.value)
        connector["finished_steps"] = finished
        return connector

    def update_request_files(self, connector: dict, new_files: str) -> dict:
        # TODO remove old file URLs if they are hosted on the plugin runner
        parsed_files = RequestFileDescriptorSchema(many=True).loads(new_files)
        connector["request_files"] = parsed_files

        finished = connector.get("finished_steps", [])
        finished.append(ConnectorKey.REQUEST_FILES.value)
        connector["finished_steps"] = finished
        return connector

    def update_response_handling(
        self, connector: dict, new_response_handling: str
    ) -> dict:
        connector["response_handling"] = new_response_handling

        finished = connector.get("finished_steps", [])
        finished.append(ConnectorKey.RESPONSE_HANDLING.value)
        connector["finished_steps"] = finished
        return connector

    def update_response_mapping(
        self, connector: dict, new_response_mapping: str
    ) -> dict:
        parsed_response_mapping = ResponseOutputSchema(many=True).loads(
            new_response_mapping
        )
        connector["response_mapping"] = parsed_response_mapping

        finished = connector.get("finished_steps", [])
        finished.append(ConnectorKey.RESPONSE_MAPPING.value)
        connector["finished_steps"] = finished
        return connector

    def deploy_as_plugin(
        self, connector: dict, connector_id: str, parent_id: str
    ) -> dict:
        if connector.get("is_deployed", False):
            return connector

        version = connector.get("version", 0) + 1
        connector["version"] = version

        connector["is_deployed"] = True
        deploy_connector(connector_id)

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

        VIRTUAL_PLUGIN_CREATED.send(
            current_app._get_current_object(), plugin_url=plugin_url
        )

        return connector

    def undeploy_plugin(
        self, connector: dict, connector_id: str, parent_id: str
    ) -> dict:
        plugin_url = url_for(
            f"{REST_CONN_BLP.name}.VirtualPluginView",
            connector_id=connector_id,
            _external=True,
        )

        if VirtualPlugin.exists([VirtualPlugin.href == plugin_url]):
            VirtualPlugin.delete_by_href(plugin_url, parent_id=parent_id)
            DB.session.commit()

        del connector["is_deployed"]

        try:
            undeploy_connector(connector_id)
        except KeyError:
            # plugin is not in the deployed connectors list
            pass

        VIRTUAL_PLUGIN_REMOVED.send(
            current_app._get_current_object(), plugin_url=plugin_url
        )
        return connector
