from http import HTTPStatus
from typing import Mapping

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
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import WORKFLOWS_BLP, DeployWorkflow
from .schemas import DeployWorkflowSchema
from .tasks import deploy_workflow

config = DeployWorkflow.instance.config

TASK_LOGGER = get_task_logger(__name__)


@WORKFLOWS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @WORKFLOWS_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata"""

        return PluginMetadata(
            title="Deploy Workflow",
            description="Deploys a BPMN workflow to Camunda and exposes it as a plugin.",
            name=DeployWorkflow.instance.name,
            version=DeployWorkflow.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{WORKFLOWS_BLP.name}.{DeployWorkflowView.__name__}"),
                ui_href=url_for(f"{WORKFLOWS_BLP.name}.{MicroFrontend.__name__}"),
                data_input=[],
                data_output=[],
            ),
            tags=DeployWorkflow.instance.tags,
        )


@WORKFLOWS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the deploy workflow plugin."""

    @WORKFLOWS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the deploy workflow plugin."
    )
    @WORKFLOWS_BLP.arguments(
        DeployWorkflowSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @WORKFLOWS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the deploy workflow plugin."
    )
    @WORKFLOWS_BLP.arguments(
        DeployWorkflowSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with pre-rendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        return Response(
            render_template(
                "simple_template.html",
                name=DeployWorkflow.instance.name,
                version=DeployWorkflow.instance.version,
                schema=DeployWorkflowSchema(),
                valid=valid,
                values=dict(data),
                errors=errors,
                process=url_for(f"{WORKFLOWS_BLP.name}.{DeployWorkflowView.__name__}"),
            )
        )


@WORKFLOWS_BLP.route("/deploy/")
class DeployWorkflowView(MethodView):
    """Deploy the workflow to camunda and as a virtual plugin."""

    @WORKFLOWS_BLP.arguments(DeployWorkflowSchema(unknown=EXCLUDE), location="form")
    @WORKFLOWS_BLP.response(HTTPStatus.SEE_OTHER)
    @WORKFLOWS_BLP.require_jwt("jwt", optional=True)
    def post(self, data: dict):
        from .management_routes import WORKFLOW_MGMNT_BLP, VirtualPluginView

        db_task = ProcessingTask(
            task_name=deploy_workflow.__name__,
            parameters=data["workflow"],
        )

        assert isinstance(db_task.data, dict)

        db_task.data["plugin_url_template"] = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            process_definition_id="{process_definition_id}",
            _external=True,
        )

        db_task.save(commit=True)

        task: chain = deploy_workflow.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
