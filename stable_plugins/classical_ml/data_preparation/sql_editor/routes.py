from http import HTTPStatus
from typing import Literal, Mapping

from flask import current_app, render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from kombu.exceptions import OperationalError
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)

from .plugin import SQL_BLP, SQLEditor
from .schemas import SQLInputSchema
from .tasks import process_sql


@SQL_BLP.route("/")
class PluginRootView(MethodView):
    """Plugin for using SQL for data processing."""

    @SQL_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @SQL_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata"""

        return PluginMetadata(
            title="SQL Editor",
            description="Use SQL to process or filter existing data.",
            name=SQLEditor.instance.name,
            version=SQLEditor.instance.version,
            type=PluginType.interaction,
            entry_point=EntryPoint(
                href=url_for(
                    f"{SQL_BLP.name}.{ProcessSQL.__name__}",
                    _external=True,
                ),
                ui_href=url_for(f"{SQL_BLP.name}.{SQLFrontend.__name__}", _external=True),
                data_input=[
                    InputDataMetadata(
                        "*",
                        content_type=["application/json", "text/csv"],
                        parameter="sql",
                        required=False,
                    )
                ],
                data_output=[],
            ),
            tags=SQLEditor.instance.tags,
        )


@SQL_BLP.route("/ui/")
class SQLFrontend(MethodView):
    """Micro frontend for the sql editor plugin."""

    @SQL_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the sql editor plugin.",
    )
    @SQL_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""

        return self.render({}, {}, True)

    @SQL_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the sql editor plugin.",
    )
    @SQL_BLP.arguments(
        SQLInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @SQL_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with pre-rendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = SQLEditor.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        schema = SQLInputSchema()
        return Response(
            render_template(
                "sql-editor.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{SQL_BLP.name}.{ProcessSQL.__name__}"),
            )
        )


@SQL_BLP.route("/process/")
class ProcessSQL(MethodView):
    """TODO."""

    @SQL_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return ""
