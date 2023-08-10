import json
from collections import ChainMap
from http import HTTPStatus
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

from celery import chain
from celery.canvas import Signature
from flask import render_template
from flask.globals import request
from flask.helpers import redirect, url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from marshmallow import EXCLUDE, INCLUDE, RAISE
from typing_extensions import Required, TypedDict

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    OutputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.virtual_plugins import PluginState, VirtualPlugin
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from .blueprint import REST_CONN_BLP
from .schemas import (
    ConnectorKey,
    ConnectorSchema,
    ConnectorUpdateSchema,
    ConnectorVariable,
    ConnectorVariableSchema,
    ConnectorVariablesInputSchema,
    RequestFileDescriptorSchema,
    ResponseOutput,
    ResponseOutputSchema,
    VariableType,
)
from .tasks import perform_request
from ..database import get_wip_connectors, save_wip_connectors, start_new_connector
from ..plugin import RESTConnector


@REST_CONN_BLP.route("/connectors/<string:connector_id>/")
class VirtualPluginView(MethodView):
    """Metadata endpoint for a virtual workflow plugin."""

    @REST_CONN_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def get(self, connector_id: str):
        plugin_url = url_for(
            f"{REST_CONN_BLP.name}.{VirtualPluginView.__name__}",
            connector_id=connector_id,
            _external=True,
        )

        plugin = VirtualPlugin.get_by_href(plugin_url, RESTConnector.instance.identifier)

        if plugin is None:
            abort(HTTPStatus.NOT_FOUND, message="Plugin does not exist.")

        parent_plugin = RESTConnector.instance
        connector = PluginState.get_value(
            parent_plugin.identifier, connector_id, default={}
        )
        assert isinstance(connector, dict), "Type assertion"

        if not connector.get("is_deployed", False):
            abort(HTTPStatus.NOT_FOUND, message="Plugin does not exist.")

        data_inputs: List[InputDataMetadata] = []
        data_outputs: List[Union[OutputDataMetadata, DataMetadata]] = []

        for var in cast(Iterable[ConnectorVariable], connector.get("variables", [])):
            if var["type"] == "data":
                data_input = InputDataMetadata(
                    data_type=var.get("data_type", "*"),
                    content_type=[var.get("content_type", "*")],
                    required=var.get("required", False),
                    parameter=var["name"],
                )
                data_inputs.append(data_input)

        for output in cast(
            Iterable[ResponseOutput], connector.get("response_mapping", [])
        ):
            data_out = OutputDataMetadata(
                data_type=output["data_type"],
                content_type=[output["content_type"]],
                name=output["name"],
                required=True,
            )
            data_outputs.append(data_out)

        return PluginMetadata(
            title="title",  # TODO get name from published connectors list
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            tags=plugin.tag_list,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(
                    f"{REST_CONN_BLP.name}.{ConnectorProcessView.__name__}",
                    connector_id=connector_id,
                    _external=True,
                ),
                ui_href=url_for(
                    f"{REST_CONN_BLP.name}.{ConnectorUiView.__name__}",
                    connector_id=connector_id,
                    _external=True,
                ),
                data_input=data_inputs,
                data_output=data_outputs,
            ),
        )


@REST_CONN_BLP.route("/connectors/<string:connector_id>/ui/")
class ConnectorUiView(MethodView):
    """UI for invoking a REST connector."""

    @REST_CONN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for a virtual rest connector plugin."
    )
    @REST_CONN_BLP.arguments(
        ConnectorVariablesInputSchema(
            partial=True, unknown=INCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, connector_id: str):
        return self.render(request.args, connector_id)

    @REST_CONN_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for a virtual rest connector plugin."
    )
    @REST_CONN_BLP.arguments(
        ConnectorVariablesInputSchema(
            partial=True, unknown=INCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, connector_id: str):
        return self.render(request.form, connector_id)

    def render(self, data: Mapping, connector_id: str):
        plugin_url = url_for(
            f"{REST_CONN_BLP.name}.{VirtualPluginView.__name__}",
            connector_id=connector_id,
            _external=True,
        )

        plugin = VirtualPlugin.get_by_href(plugin_url, RESTConnector.instance.identifier)

        if plugin is None:
            abort(HTTPStatus.NOT_FOUND, message="Plugin does not exist.")

        parent_plugin = RESTConnector.instance
        connector = PluginState.get_value(
            parent_plugin.identifier, connector_id, default={}
        )
        assert isinstance(connector, dict), "Type assertion"

        schema = ConnectorVariablesInputSchema(
            cast(List[ConnectorVariable], connector.get("variables", []))
        )

        errors = schema.validate(data, partial=True)

        data = dict(data)

        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name if plugin else "UNKNOWN",
                version=plugin.version if plugin else "-1",
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{REST_CONN_BLP.name}.{ConnectorProcessView.__name__}",
                    connector_id=connector_id,
                ),
            )
        )


@REST_CONN_BLP.route("/connectors/<string:connector_id>/process/")
class ConnectorProcessView(MethodView):
    """TODO"""

    @REST_CONN_BLP.arguments(
        ConnectorVariablesInputSchema(partial=True, unknown=INCLUDE), location="form"
    )
    @REST_CONN_BLP.response(HTTPStatus.SEE_OTHER)
    @REST_CONN_BLP.require_jwt("jwt", optional=True)
    def post(self, data: Mapping, connector_id: str):
        parent_plugin = RESTConnector.instance
        connector = PluginState.get_value(
            parent_plugin.identifier, connector_id, default={}
        )
        assert isinstance(connector, dict), "Type assertion"

        schema = ConnectorVariablesInputSchema(
            cast(List[ConnectorVariable], connector.get("variables", []))
        )

        validated_data = schema.load(data)

        db_task = ProcessingTask(
            task_name=perform_request.name,
            parameters=json.dumps(validated_data),
        )
        db_task.save(commit=False)
        DB.session.flush()  # flsuh to DB to get db_task id populated

        db_task.save(commit=True)

        task: Union[chain, Signature] = perform_request.s(
            connector_id=connector_id, db_id=db_task.id
        ) | save_task_result.s(
            db_id=db_task.id
        )  # TODO save task results in chain (and adjust type signature)
        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
