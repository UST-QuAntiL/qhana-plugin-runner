from http import HTTPStatus
from json import dumps
from typing import Mapping

from celery.canvas import chain
import celery
from flask import Response, current_app, jsonify, redirect, render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask_smorest import abort
from marshmallow import EXCLUDE
from werkzeug.exceptions import HTTPException

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from .plugin import SQL_BLP, SQLEditor
from .schemas import SQLInputSchema
from .tasks import preview_sql, process_sql
from .util import PREVIEW_LIMIT, execute_sql, serialize_rows


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
            type=PluginType.processing,
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
                data_output=[
                    DataMetadata(
                        data_type="*",
                        content_type=["text/csv", "application/json"],
                        required=True,
                    )
                ],
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

        render_errors = dict(errors) if errors else {}

        schema = SQLInputSchema()
        return Response(
            render_template(
                "sql-editor.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=render_errors,
                process=url_for(f"{SQL_BLP.name}.{ProcessSQL.__name__}"),
                check=url_for(f"{SQL_BLP.name}.{CheckSQL.__name__}"),
                preview=url_for(f"{SQL_BLP.name}.{PreviewSQL.__name__}"),
            )
        )


@SQL_BLP.route("/check/")
class CheckSQL(MethodView):
    """Endpoint to validate SQL syntax and basic safety."""

    @SQL_BLP.arguments(SQLInputSchema(unknown=EXCLUDE), location="form")
    @SQL_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        return Response(status=HTTPStatus.NO_CONTENT)


@SQL_BLP.route("/preview/")
class PreviewSQL(MethodView):
    """Endpoint to preview SQL output."""

    @SQL_BLP.arguments(SQLInputSchema(unknown=EXCLUDE), location="form")
    @SQL_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        sql = arguments.get("sql", "")
        use_celery = current_app.config.get("SQL_EDITOR_PREVIEW_USE_CELERY", False)
        preview_timeout = current_app.config.get("SQL_EDITOR_PREVIEW_TIMEOUT_SEC", 10)
        try:
            if use_celery:
                task_result = preview_sql.s(sql, PREVIEW_LIMIT).apply_async()
                try:
                    payload = task_result.get(timeout=preview_timeout)
                except celery.exceptions.TimeoutError:
                    task_result.revoke(terminate=True)
                    abort(
                        HTTPStatus.REQUEST_TIMEOUT,
                        message=f"Preview timed out after {preview_timeout} seconds.",
                    )
                columns = payload.get("columns", [])
                serialized_rows = payload.get("rows", [])
            else:
                columns, rows = execute_sql(sql, limit=PREVIEW_LIMIT)
                serialized_rows = list(serialize_rows(rows))
        except HTTPException:
            raise
        except ValueError as err:
            return (
                jsonify({"ok": False, "error": str(err)}),
                HTTPStatus.BAD_REQUEST,
            )
        except Exception as err:
            return (
                jsonify({"ok": False, "error": str(err)}),
                HTTPStatus.BAD_REQUEST,
            )

        return jsonify(
            {
                "ok": True,
                "columns": columns,
                "rows": serialized_rows,
                "limit": PREVIEW_LIMIT,
                "row_count": len(serialized_rows),
                "truncated": len(serialized_rows) == PREVIEW_LIMIT,
            }
        )


@SQL_BLP.route("/process/")
class ProcessSQL(MethodView):
    """Start a long running processing task."""

    @SQL_BLP.arguments(SQLInputSchema(unknown=EXCLUDE), location="form")
    @SQL_BLP.response(HTTPStatus.SEE_OTHER)
    @SQL_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        sql = arguments.get("sql", "")

        db_task = ProcessingTask(task_name=process_sql.name, parameters=dumps(arguments))
        db_task.save(commit=True)

        task: chain = process_sql.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
