from http import HTTPStatus
from logging import Logger
from typing import Mapping, Optional

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import INTERACTION_DEMO_BLP, InteractionDemo
from .schemas import InputParametersSchema, TaskResponseSchema
from .tasks import processing_task_1, processing_task_2
from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result


@INTERACTION_DEMO_BLP.route("/")
class MetadataView(MethodView):
    """Plugin metadata resource."""

    @INTERACTION_DEMO_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=InteractionDemo.instance.name,
            description=INTERACTION_DEMO_BLP.description,
            name=InteractionDemo.instance.identifier,
            version=InteractionDemo.instance.version,
            type=PluginType.complex,
            entry_point=EntryPoint(
                href=url_for(f"{INTERACTION_DEMO_BLP.name}.ProcessStep1View"),
                ui_href=url_for(f"{INTERACTION_DEMO_BLP.name}.MicroFrontendStep1"),
                data_input=[],
                data_output=[],
            ),
            tags=[],
        )


@INTERACTION_DEMO_BLP.route("/ui-step-1/")
class MicroFrontendStep1(MethodView):
    """Micro frontend for step 1 of the interaction demo plugin."""

    example_inputs = {
        "inputStr": "test1",
    }

    @INTERACTION_DEMO_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for step 1 of the interaction demo plugin.",
    )
    @INTERACTION_DEMO_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @INTERACTION_DEMO_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for step 1 of the interaction demo plugin.",
    )
    @INTERACTION_DEMO_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=InteractionDemo.instance.name,
                version=InteractionDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{INTERACTION_DEMO_BLP.name}.ProcessStep1View"),
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.MicroFrontendStep1",
                    **self.example_inputs,
                ),
            )
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@INTERACTION_DEMO_BLP.route("/process-step-1/")
class ProcessStep1View(MethodView):
    """Start a long running processing task."""

    @INTERACTION_DEMO_BLP.arguments(
        InputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @INTERACTION_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name=processing_task_1.name,
        )
        db_task.save(commit=True)

        db_task.data["input_str"] = arguments["input_str"]
        db_task.data["href"] = url_for(
            f"{INTERACTION_DEMO_BLP.name}.ProcessStep2View",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["ui_href"] = url_for(
            f"{INTERACTION_DEMO_BLP.name}.MicroFrontendStep2",
            db_id=db_task.id,
            _external=True,
        )
        db_task.save(commit=True)

        # next step
        step_id = "demo-step"
        href = url_for(
            f"invokable-demo@v0-1-0.ProcessView", db_id=db_task.id, _external=True
        )  # FIXME replace hardcoded plugin name with user input
        ui_href = url_for(
            f"invokable-demo@v0-1-0.MicroFrontend", db_id=db_task.id, _external=True
        )  # FIXME replace hardcoded plugin name with user input

        # all tasks need to know about db id to load the db entry
        task: chain = processing_task_1.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=33
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@INTERACTION_DEMO_BLP.route("/<int:db_id>/ui-step-2/")
class MicroFrontendStep2(MethodView):
    """Micro frontend for step 2 of the interaction demo plugin."""

    example_inputs = {
        "inputStr": "test3",
    }

    @INTERACTION_DEMO_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for step 2 of the interaction demo plugin.",
    )
    @INTERACTION_DEMO_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @INTERACTION_DEMO_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for step 2 of the interaction demo plugin.",
    )
    @INTERACTION_DEMO_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        # retrieve input data from preprocessing
        if not data:
            try:
                input_str = db_task.data["input_str"]
            except:
                input_str = ""
            data = {"inputStr": "Input from preprocessing: " + input_str}

        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=InteractionDemo.instance.name,
                version=InteractionDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.ProcessStep2View", db_id=db_id
                ),
                example_values=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.MicroFrontendStep2",
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@INTERACTION_DEMO_BLP.route("/<int:db_id>/process-step-2/")
class ProcessStep2View(MethodView):
    """Start a long running processing task."""

    @INTERACTION_DEMO_BLP.arguments(
        InputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @INTERACTION_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the demo task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.data["input_str"] = arguments["input_str"]
        db_task.clear_previous_step()
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = processing_task_2.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )