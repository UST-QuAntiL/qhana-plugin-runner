import json
from http import HTTPStatus
from typing import Mapping, Optional

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import redirect, render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error

from . import WORKFLOWS_BLP, Workflows
from .schemas import (
    AnyInputSchema,
    InputParameters,
    WorkflowsParametersSchema,
    WorkflowsTaskResponseSchema,
)
from .tasks import process_input, start_workflow
from .watchers.human_task_watcher import human_task_watcher

config = Workflows.instance.config

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
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{WORKFLOWS_BLP.name}.ProcessView"),
                ui_href=url_for(f"{WORKFLOWS_BLP.name}.MicroFrontend"),
                data_input=[],
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
            parameters=WorkflowsParametersSchema().dumps(input_params),
        )
        db_task.save(commit=False)
        DB.session.flush()  # flsuh to DB to get db_task id populated

        assert isinstance(db_task.data, dict)

        db_task.data["href"] = url_for(
            f"{WORKFLOWS_BLP.name}.{HumanTaskProcessView.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["ui_href"] = url_for(
            f"{WORKFLOWS_BLP.name}.{HumanTaskFrontend.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        db_task.save(commit=True)

        task: chain = start_workflow.s(db_id=db_task.id) | human_task_watcher.si(
            db_id=db_task.id
        )
        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@WORKFLOWS_BLP.route("/<int:db_id>/human-task-ui/")
class HumanTaskFrontend(MethodView):
    """Micro frontend of a workflow human task."""

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

        form_params = None

        if not data:
            assert isinstance(db_task.data, dict)
            try:
                form_params = json.loads(db_task.data["form_params"])
            except:
                form_params = None  # TODO raise proper error here (will throw generic one two lines later)

        data = {}
        for key, val in form_params.items():
            prefix_file_url = config["qhana_input"]["prefix_value_file_url"]
            prefix_delimiter = config["qhana_input"]["prefix_value_delimiter"]
            if val["value"]:
                if not (
                    val["type"] == "String"
                    and val["value"].startswith(f"{prefix_file_url}{prefix_delimiter}")
                ):
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
                process=url_for(
                    f"{WORKFLOWS_BLP.name}.HumanTaskProcessView", db_id=db_id
                ),
            )
        )


@WORKFLOWS_BLP.route("/<int:db_id>/human-task-ui/bpmn_io")
class HumanTaskBPMNVisualizationFrontend(MethodView):
    """Micro frontend for bpmn io."""

    @WORKFLOWS_BLP.html_response(HTTPStatus.OK, description="Micro frontend for bpmn io.")
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

    @WORKFLOWS_BLP.html_response(HTTPStatus.OK, description="Micro frontend for bpmn io.")
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

        bpmn_properties = None

        if not data:
            try:
                assert isinstance(db_task.data, dict)
                bpmn_properties = json.loads(db_task.data["bpmn"])  # FIXME!!!
            except:
                bpmn_properties = None

        return Response(
            render_template(
                "bpmn_io.html",
                values=bpmn_properties,
            )
        )


@WORKFLOWS_BLP.route("/<int:db_id>/human-task-process/")
class HumanTaskProcessView(MethodView):
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
        task: chain = process_input.s(db_id=db_task.id) | human_task_watcher.si(
            db_id=db_task.id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
