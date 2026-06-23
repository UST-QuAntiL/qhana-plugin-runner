from http import HTTPStatus
from json import dumps, loads
from typing import Mapping, Optional

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import current_app, request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import (
    TASK_STEPS_CHANGED,
    add_step,
    save_task_error,
    save_task_result,
)

from . import TAX_ROUTER_BLP, TaxonomyRouter
from .schemas import (
    PIPELINE_OPTIONS,
    RoutingStepParametersSchema,
    TaxonomyRouterParametersSchema,
)
from .tasks import preprocessing_task, processing_task

TASK_LOGGER = get_task_logger(__name__)


@TAX_ROUTER_BLP.route("/")
class PluginsView(MethodView):
    @TAX_ROUTER_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @TAX_ROUTER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = TaxonomyRouter.instance
        return PluginMetadata(
            title="Taxonomy Router",
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{TAX_ROUTER_BLP.name}.ProcessView"),
                ui_href=url_for(f"{TAX_ROUTER_BLP.name}.MicroFrontend"),
                data_input=[],  # TODO
                data_output=[],  # TODO
            ),
            tags=plugin.tags,
        )


@TAX_ROUTER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    @TAX_ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the taxonomy router plugin."
    )
    @TAX_ROUTER_BLP.arguments(
        TaxonomyRouterParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @TAX_ROUTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @TAX_ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the taxonomy router plugin."
    )
    @TAX_ROUTER_BLP.arguments(
        TaxonomyRouterParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @TAX_ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = TaxonomyRouter.instance

        return Response(
            render_template(
                "simple_template.html",
                name=plugin.name,
                version=plugin.version,
                schema=TaxonomyRouterParametersSchema(),
                values=data,
                valid=valid,
                errors=errors,
                process=url_for(f"{TAX_ROUTER_BLP.name}.ProcessView"),
            )
        )


@TAX_ROUTER_BLP.route("/process/")
class ProcessView(MethodView):
    @TAX_ROUTER_BLP.arguments(
        TaxonomyRouterParametersSchema(unknown=EXCLUDE), location="form"
    )
    @TAX_ROUTER_BLP.response(HTTPStatus.SEE_OTHER)
    @TAX_ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the preprocessing task and queue the routing step."""
        db_task = ProcessingTask(task_name="taxonomy-router", parameters=dumps(arguments))
        db_task.save(commit=True)

        step_id = "routing-step"
        href = url_for(
            f"{TAX_ROUTER_BLP.name}.RoutingStepView", db_id=db_task.id, _external=True
        )
        ui_href = url_for(
            f"{TAX_ROUTER_BLP.name}.RoutingStepFrontend",
            db_id=db_task.id,
            _external=True,
        )

        task: chain = preprocessing_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=50
        )

        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@TAX_ROUTER_BLP.route("/<int:db_id>/routing-step-ui/")
class RoutingStepFrontend(MethodView):
    """Micro frontend for the routing step (one pipeline dropdown per attribute)."""

    @TAX_ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the taxonomy router routing step."
    )
    @TAX_ROUTER_BLP.arguments(
        RoutingStepParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @TAX_ROUTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors, False)

    @TAX_ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the taxonomy router routing step."
    )
    @TAX_ROUTER_BLP.arguments(
        RoutingStepParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @TAX_ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors, not errors)

    def render(self, data: Mapping, db_id: int, errors: dict, valid: bool):
        plugin = TaxonomyRouter.instance
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        attributes = db_task.data.get("taxonomy_attributes", [])
        input_params = loads(db_task.parameters or "{}")

        return Response(
            render_template(
                "routing_step.html",
                name=plugin.name,
                version=plugin.version,
                schema=RoutingStepParametersSchema(),
                attributes=attributes,
                pipeline_options=PIPELINE_OPTIONS,
                input_params=input_params,
                values=data,
                valid=valid,
                errors=errors,
                process=url_for(f"{TAX_ROUTER_BLP.name}.RoutingStepView", db_id=db_id),
            )
        )


@TAX_ROUTER_BLP.route("/<int:db_id>/routing-step-process/")
class RoutingStepView(MethodView):
    """Record the routing selection and finish the task."""

    @TAX_ROUTER_BLP.arguments(
        RoutingStepParametersSchema(unknown=EXCLUDE), location="form"
    )
    @TAX_ROUTER_BLP.response(HTTPStatus.SEE_OTHER)
    @TAX_ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the final processing task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        # Merge the routing selections into the step 1 parameters (the entity, metadata, and taxonomy URLs) so both are available downstream.
        parameters = loads(db_task.parameters or "{}")
        parameters.update(arguments)
        db_task.parameters = dumps(parameters)
        db_task.clear_previous_step()
        db_task.save(commit=True)

        app = current_app._get_current_object()
        TASK_STEPS_CHANGED.send(app, task_id=db_id)

        task: chain = processing_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
