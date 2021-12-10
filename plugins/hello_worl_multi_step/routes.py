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

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginType,
    EntryPoint,
)

from plugins.hello_worl_multi_step import HELLO_MULTI_BLP, HelloWorldMultiStep
from plugins.hello_worl_multi_step.schemas import (
    DemoResponseSchema,
    HelloWorldParametersSchema,
    TaskResponseSchema,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import (
    add_step,
    save_task_error,
    save_task_result,
)

from plugins.hello_worl_multi_step.tasks import preprocessing_task, processing_task


@HELLO_MULTI_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @HELLO_MULTI_BLP.response(HTTPStatus.OK, DemoResponseSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=HelloWorldMultiStep.instance.name,
            description=HELLO_MULTI_BLP.description,
            name=HelloWorldMultiStep.instance.identifier,
            version=HelloWorldMultiStep.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href="./process/", ui_href="./ui/", data_input=[], data_output=[]
            ),
            tags=[],
        )


@HELLO_MULTI_BLP.route("/ui/")
class MicroFrontend(MethodView):
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
        """Start the data preprocessing task."""
        db_task = ProcessingTask(task_name="manual-classification")
        db_task.save(commit=True)

        schema = HelloWorldParametersSchema()
        return Response(
            render_template(
                "hello_template.html",
                name=HelloWorldMultiStep.instance.name,
                version=HelloWorldMultiStep.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{HELLO_MULTI_BLP.name}.ProcessView", db_id=db_task.id),
                example_values=url_for(
                    f"{HELLO_MULTI_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


TASK_LOGGER = get_task_logger(__name__)


@HELLO_MULTI_BLP.route("/<string:db_id>/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(unknown=EXCLUDE), location="form"
    )
    @HELLO_MULTI_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: str):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)
        db_task.parameters = dumps(arguments)
        db_task.save(commit=True)

        # next step
        step_id = "step1"
        href = url_for(f"{HELLO_MULTI_BLP.name}.Step1Frontend", db_id=db_task.id)
        ui_href = url_for(f"{HELLO_MULTI_BLP.name}.Step1View", db_id=db_task.id)

        # all tasks need to know about db id to load the db entry
        task: chain = preprocessing_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=50
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@HELLO_MULTI_BLP.route("/<string:db_id>/step1-ui/")
class Step1Frontend(MethodView):
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
    def get(self, errors, db_id: str):
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
    def post(self, errors, db_id: str):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: str, errors: dict):
        # TODO: retrieve and display data
        schema = HelloWorldParametersSchema()
        return Response(
            render_template(
                "hello_template.html",
                name=HelloWorldMultiStep.instance.name,
                version=HelloWorldMultiStep.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{HELLO_MULTI_BLP.name}.Step1View", db_id=db_id),
                example_values=url_for(
                    f"{HELLO_MULTI_BLP.name}.Step1Frontend",
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@HELLO_MULTI_BLP.route("/<string:db_id>/step1/")
class Step1View(MethodView):
    """Start a long running processing task."""

    @HELLO_MULTI_BLP.arguments(
        HelloWorldParametersSchema(unknown=EXCLUDE), location="form"
    )
    @HELLO_MULTI_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @HELLO_MULTI_BLP.require_jwt("jwt", optional=True)
    def post(self, error, db_id: str):
        """Start the demo task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

        # set previous step to cleared
        db_task.clear_previous_step()

        # all tasks need to know about db id to load the db entry
        task: chain = processing_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", db_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
