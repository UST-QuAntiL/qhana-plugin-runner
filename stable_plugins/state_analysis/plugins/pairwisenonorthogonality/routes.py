from http import HTTPStatus
from json import dumps
from typing import Mapping

from celery.canvas import chain
from flask import Response, abort, redirect, render_template, request, url_for
from flask.views import MethodView
from marshmallow import EXCLUDE
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    OutputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import BLP, Plugin
from .schemas import Schema
from .tasks import task

vectors_example = [
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [1.0, 0.0]],
    [[1.0, 0.0], [1.0, 0.0]],
]
_exampleInputs_ = {
    "vectors": f"{vectors_example}",
    "tolerance": "1e-10",
}

_description_ = "Pairwise orthogonality plugin UI"

_data_input_ = [
    DataMetadata(
        data_type="application/json",
        content_type=["application/json"],
        required=True,
    )
]
_data_output_ = [
    OutputDataMetadata(
        data_type="custom/pairwise-orthogonality-output",
        content_type=["application/json"],
        required=True,
        name="pairwise-orthogonality-output.json",
    )
]

_help_text_ = "Check if all vectors are pairwise orthogonal or decode from circuit. True if all pairs orthogonal."

_task_ = task


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
                plugin_dependencies=[],
                data_input=_data_input_,
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
            task_name=_task_.name,
            parameters=dumps(arguments),
        )
        db_task.save(commit=True)

        task: chain = _task_.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
