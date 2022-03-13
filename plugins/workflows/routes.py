from http import HTTPStatus
from json import dumps
from typing import Mapping
from celery.result import AsyncResult
from flask import Response, render_template, redirect
from flask.views import MethodView
from marshmallow import EXCLUDE
from flask.globals import request
from flask.helpers import url_for
from celery.canvas import chain
from plugins.workflows import WORKFLOWS_BLP, Workflows
from plugins.workflows.schemas import WorkflowsResponseSchema, WorkflowsParametersSchema, WorkflowsTaskResponseSchema, \
    InputParameters
from plugins.workflows.tasks import run_model
from qhana_plugin_runner.api.plugin_schemas import PluginMetadataSchema, PluginMetadata, PluginType, EntryPoint, \
    DataMetadata
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.db.models.tasks import ProcessingTask


@WORKFLOWS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @WORKFLOWS_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata"""

        return PluginMetadata(
            title="Workflows",
            description="Runs workflows",
            name=Workflows.instance.name,
            version=Workflows.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{WORKFLOWS_BLP.name}.ProcessView"),
                ui_href=url_for(f"{WORKFLOWS_BLP.name}.MicroFrontend"),
                data_input=[
                    DataMetadata(
                        data_type="string",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
                data_output=[],
            ),
            tags=["bpmn", "camunda engine"],
        )

@WORKFLOWS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the workflows plugin."""

    @WORKFLOWS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the workflows plugin."
    )
    @WORKFLOWS_BLP.arguments(
        WorkflowsParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @WORKFLOWS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the workflows plugin."
    )
    @WORKFLOWS_BLP.arguments(
        WorkflowsParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with pre-rendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        return Response(
            render_template(
                "simple_template.html",
                name=Workflows.instance.name,
                version=Workflows.instance.version,
                schema=WorkflowsParametersSchema(),
                values=dict(data),
                errors=errors,
                process=url_for(f"{WORKFLOWS_BLP.name}.ProcessView"),
            )
        )


@WORKFLOWS_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @WORKFLOWS_BLP.arguments(WorkflowsParametersSchema(unknown=EXCLUDE), location="form")
    @WORKFLOWS_BLP.response(HTTPStatus.OK, WorkflowsTaskResponseSchema())
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def post(self, input_params: InputParameters):

        db_task = ProcessingTask(
            task_name=run_model.name,
            parameters=WorkflowsParametersSchema().dumps(input_params)
        )
        db_task.save(commit=True)

        task: chain = run_model.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )





