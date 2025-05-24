from http import HTTPStatus
from json import dumps
from typing import Mapping, Optional

from celery.canvas import chain
from flask import Response, abort, redirect, render_template, request, url_for
from flask.views import MethodView
from marshmallow import EXCLUDE, fields

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    MaBaseSchema,
    OutputDataMetadata,
    PluginDependencyMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import (
    TASK_DETAILS_CHANGED,
    TASK_STEPS_CHANGED,
    save_task_error,
    save_task_result,
)

from . import BLP, Plugin
from .schemas import Schema
from .tasks import building_circuit_and_simulate_task, get_restults_task


class WebhookParams(MaBaseSchema):
    source = fields.URL()
    event = fields.String()


_exampleInputs_ = {}

_description_ = "Pairwise orthogonality plugin UI"

_data_output_ = [
    OutputDataMetadata(
        data_type="custom/quantum-orthogonality-output",
        content_type=["application/json"],
        required=True,
        name="quantum-orthogonality-output.json",
    )
]

_help_text_ = (
    "Provide a set of complex vectors and a tolerance. "
    "The plugin checks whether they are pairwise orthogonal within the given tolerance."
)

_building_circuit_and_simulate_task_ = building_circuit_and_simulate_task


@BLP.route("/")
class PluginView(MethodView):
    """Returns plugin metadata."""

    @BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    def get(self):
        plugin = Plugin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{BLP.name}.ProcessView"),
                ui_href=url_for(f"{BLP.name}.MicroFrontend"),
                plugin_dependencies=[
                    PluginDependencyMetadata(
                        required=True,
                        parameter="executor",
                        tags=["circuit-executor", "qasm-2"],
                    ),
                ],
                data_input=[],
                data_output=_data_output_,
            ),
            tags=plugin.tags,
        )


@BLP.route("/ui/")
class MicroFrontend(MethodView):
    """A basic UI"""

    example_inputs = _exampleInputs_

    @BLP.html_response(HTTPStatus.OK, description=_description_ + " (GET).")
    @BLP.arguments(
        Schema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    def get(self, errors):
        return self.render(request.args, errors, valid=False)

    @BLP.html_response(HTTPStatus.OK, description=_description_ + " (POST).")
    @BLP.arguments(
        Schema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    def post(self, errors):
        return self.render(request.form, errors, valid=(not errors))

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = Plugin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        schema = Schema()
        result = None
        task_id = data.get("task_id")
        if task_id:
            task = ProcessingTask.get_by_id(task_id)
            if task:
                result = task.result

        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                result=result,
                process=url_for(f"{BLP.name}.ProcessView"),
                help_text=_help_text_,
                example_values=url_for(
                    f"{BLP.name}.MicroFrontend",
                    **self.example_inputs,
                ),
            )
        )


@BLP.route("/process/")
class ProcessView(MethodView):
    """
    Starts task.
    """

    @BLP.arguments(
        Schema(unknown=EXCLUDE),
        location="form",
    )
    @BLP.response(HTTPStatus.SEE_OTHER)
    def post(self, arguments):

        db_task = ProcessingTask(
            task_name=_building_circuit_and_simulate_task_.name,
            parameters=Schema().dumps(arguments),
        )
        db_task.save()
        DB.session.flush()
        continue_url = url_for(
            f"{BLP.name}.{ContinueProcessView.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        db_task.data = {
            "continue_url": continue_url,
        }
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task = _building_circuit_and_simulate_task_.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@BLP.route("/continue/<int:db_id>/")
class ContinueProcessView(MethodView):
    """Restart long running task that was blocked by an ongoing plugin computation."""

    @BLP.arguments(WebhookParams(partial=True), location="query")
    @BLP.response(HTTPStatus.NO_CONTENT)
    def post(self, params: dict, db_id: int):
        """Check for updates in plugin computation and resume processing."""
        task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if task_data is None:
            abort(HTTPStatus.NOT_FOUND)

        if task_data.task_name != _building_circuit_and_simulate_task_.name:
            # processing task is from another plugin, cannot resume
            abort(HTTPStatus.NOT_FOUND)

        if not isinstance(task_data.data, dict):
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        event_source = params.get("source", None)
        event_type = params.get("event", None)

        result_url = task_data.data.get("result_url")

        if event_source != result_url:
            abort(HTTPStatus.NOT_FOUND)

        if not result_url or task_data.is_finished:
            abort(HTTPStatus.NOT_FOUND)

        task = get_restults_task.s(db_id=db_id, event_type=event_type)
        task.link_error(save_task_error.s(db_id=db_id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
