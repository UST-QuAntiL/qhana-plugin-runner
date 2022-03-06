from http import HTTPStatus
from json import dumps
from typing import Mapping, Optional

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from plugins.hello_worl_multi_step import HELLO_MULTI_BLP, HelloWorldMultiStep
from plugins.hello_worl_multi_step.schemas import (
    HelloWorldParametersSchema,
    TaskResponseSchema,
)
from plugins.hello_worl_multi_step.tasks import preprocessing_task, processing_task
from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result


@HELLO_MULTI_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @HELLO_MULTI_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=HelloWorldMultiStep.instance.name,
            description=HELLO_MULTI_BLP.description,
            name=HelloWorldMultiStep.instance.identifier,
            version=HelloWorldMultiStep.instance.version,
            type=PluginType.complex,
            entry_point=EntryPoint(
                href="./process/", ui_href="./ui/", data_input=[], data_output=[]
            ),
            tags=[],
        )


@HELLO_MULTI_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the hello world plugin."""

    example_inputs = {
        "inputStr": "Test.",
    }

    @HELLO_MULTI_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @HELLO_MULTI_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = HelloWorldParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=HelloWorldMultiStep.instance.name,
                version=HelloWorldMultiStep.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{HELLO_MULTI_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{HELLO_MULTI_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


TASK_LOGGER = get_task_logger(__name__)


@HELLO_MULTI_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(unknown=EXCLUDE), location="form"
    )
    @HELLO_MULTI_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name="manual-classification", parameters=dumps(arguments)
        )
        db_task.save(commit=True)

        # next step
        step_id = "demo-step"
        href = url_for(
            f"{HELLO_MULTI_BLP.name}.DemoStepView", db_id=db_task.id, _external=True
        )
        ui_href = url_for(
            f"{HELLO_MULTI_BLP.name}.DemoStepFrontend", db_id=db_task.id, _external=True
        )

        # all tasks need to know about db id to load the db entry
        task: chain = preprocessing_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=50
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@HELLO_MULTI_BLP.route("/<int:db_id>/demo-step-ui/")
class DemoStepFrontend(MethodView):
    """Micro frontend for the hello world plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @HELLO_MULTI_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @HELLO_MULTI_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hello world plugin."
    )
    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
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

        schema = HelloWorldParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=HelloWorldMultiStep.instance.name,
                version=HelloWorldMultiStep.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{HELLO_MULTI_BLP.name}.DemoStepView", db_id=db_id),
                example_values=url_for(
                    f"{HELLO_MULTI_BLP.name}.DemoStepFrontend",
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@HELLO_MULTI_BLP.route("/<int:db_id>/demo-step-process/")
class DemoStepView(MethodView):
    """Start a long running processing task."""

    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(unknown=EXCLUDE), location="form"
    )
    @HELLO_MULTI_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the demo task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = dumps(arguments)
        db_task.clear_previous_step()
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = processing_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
