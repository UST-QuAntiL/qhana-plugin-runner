import ast
import json
from http import HTTPStatus
from typing import Mapping, Optional
from celery.utils.log import get_task_logger
from flask import Response, render_template, redirect
from flask.views import MethodView
from marshmallow import EXCLUDE
from flask.globals import request, current_app
from flask.helpers import url_for
from celery.canvas import chain
from plugins.workflows import Workflows, WORKFLOWS_BLP, conf as config
from plugins.workflows.schemas import WorkflowsParametersSchema, WorkflowsTaskResponseSchema, InputParameters, \
    AnyInputSchema
from plugins.workflows.tasks import process_input, start_workflow
from qhana_plugin_runner.api.plugin_schemas import PluginMetadataSchema, PluginMetadata, PluginType, EntryPoint, \
    DataMetadata
from qhana_plugin_runner.tasks import save_task_error
from qhana_plugin_runner.db.models.tasks import ProcessingTask

TASK_LOGGER = get_task_logger(__name__)


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
            type=PluginType.complex,
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
            task_name=start_workflow.name,
            parameters=WorkflowsParametersSchema().dumps(input_params)
        )
        db_task.save(commit=True)
        task: chain = start_workflow.s(db_id=db_task.id)
        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@WORKFLOWS_BLP.route("/<int:db_id>/human-task-ui/")
class DemoStepFrontend(MethodView):
    """Micro frontend for human tasks."""

    @WORKFLOWS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of a workflow human task."
    )
    @WORKFLOWS_BLP.arguments(
        WorkflowsParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @WORKFLOWS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of a workflow human task."
    )
    @WORKFLOWS_BLP.arguments(
        WorkflowsParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if not data:
            try:
                form_params = ast.literal_eval(db_task.data["form_params"])
            except:
                form_params = None

        # TODO: Cleanup
        data = {}
        for key, val in form_params.items():
            prefix_file_url = config["qhana_input"]["prefix_value_file_url"]
            prefix_delimiter = config["qhana_input"]["prefix_value_delimiter"]
            if val["value"]:
                if not (val["type"] == "String" and val["value"].startswith(f"{prefix_file_url}{prefix_delimiter}")):
                    data[key] = val["value"]

        schema = AnyInputSchema(form_params)

        return Response(
            render_template(
                "workflows_template.html",
                name=Workflows.instance.name,
                version=Workflows.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{WORKFLOWS_BLP.name}.DemoStepView", db_id=db_id),
            )
        )


@WORKFLOWS_BLP.route("/<int:db_id>/human-task-ui/bpmn_io")
class DemoStepFrontend(MethodView):
    """Micro frontend for bpmn io."""

    @WORKFLOWS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for bpmn io."
    )
    @WORKFLOWS_BLP.arguments(
        WorkflowsParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @WORKFLOWS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for bpmn io."
    )
    @WORKFLOWS_BLP.arguments(
        WorkflowsParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        if not data:
            try:
                bpmn_properties = ast.literal_eval(db_task.data["bpmn"])
            except:
                bpmn_properties = None

        return Response(
            render_template(
                "bpmn_io.html",
                values=bpmn_properties,
            )
        )


@WORKFLOWS_BLP.route("/<int:db_id>/human-task-process/")
class DemoStepView(MethodView):
    """Start a long running processing task."""

    @WORKFLOWS_BLP.arguments(AnyInputSchema(), location="form")
    @WORKFLOWS_BLP.response(HTTPStatus.OK, WorkflowsTaskResponseSchema())
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = json.dumps(arguments)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = process_input.s(db_id=db_task.id)

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
