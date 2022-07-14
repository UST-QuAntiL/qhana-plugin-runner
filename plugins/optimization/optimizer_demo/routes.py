from http import HTTPStatus
from json import dumps
from logging import Logger
from typing import Mapping, Optional
from urllib.parse import urljoin

import requests
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    DataMetadata,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import add_step, save_task_error, save_task_result
from . import INTERACTION_DEMO_BLP, OptimizerDemo
from .schemas import (
    InputParametersSchema,
    TaskResponseSchema,
    DatasetInputSchema,
    CallbackSchema,
)
from .tasks import no_op_task, processing_task_2


@INTERACTION_DEMO_BLP.route("/")
class MetadataView(MethodView):
    """Plugin metadata resource."""

    @INTERACTION_DEMO_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=OptimizerDemo.instance.name,
            description=INTERACTION_DEMO_BLP.description,
            name=OptimizerDemo.instance.identifier,
            version=OptimizerDemo.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.ObjFuncSetupProcess"
                ),  # URL for the first process endpoint
                ui_href=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.ObjFuncSelectionUI"
                ),  # URL for the first micro frontend endpoint
                interaction_endpoints=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    ),
                ],
            ),
            tags=[],
        )


@INTERACTION_DEMO_BLP.route("/obj-func-selection/")
class ObjFuncSelectionUI(MethodView):
    """Micro frontend for the selection of the objective function plugin."""

    # FIXME: remove when plugin selection PR has been merged
    example_inputs = {
        "objectiveFunctionUrl": "http://localhost:5005/plugins/objective-function-demo%40v0-1-0",
    }

    @INTERACTION_DEMO_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the objective function plugin.",
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
        description="Micro frontend for the selection of the objective function plugin.",
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
                name=OptimizerDemo.instance.name,
                version=OptimizerDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.ObjFuncSetupProcess"
                ),  # URL of the first processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.ObjFuncSelectionUI",  # URL of this endpoint
                    **self.example_inputs,
                ),
            )
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@INTERACTION_DEMO_BLP.route("/obj-func-setup/")
class ObjFuncSetupProcess(MethodView):
    """Start the processing task of step 1."""

    @INTERACTION_DEMO_BLP.arguments(
        InputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @INTERACTION_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the objective function setup task."""
        db_task = ProcessingTask(
            task_name=no_op_task.name,
        )
        db_task.save(commit=True)

        objective_function_url = arguments["objective_function_url"]
        db_task.data["objective_function_url"] = objective_function_url
        db_task.save(commit=True)

        # add new step where the setup of the objective function plugin is executed

        # name of the next step
        step_id = "objective function setup"

        schema = PluginMetadataSchema()
        raw_metadata = requests.get(objective_function_url).json()
        plugin_metadata: PluginMetadata = schema.load(raw_metadata)
        # URL of the process endpoint of the invoked plugin
        href = urljoin(objective_function_url, plugin_metadata.entry_point.href)
        # URL of the micro frontend endpoint of the invoked plugin
        ui_href = urljoin(objective_function_url, plugin_metadata.entry_point.ui_href)

        # Chain the first processing task with executing the objective function plugin.
        # All tasks use the same db_id to be able to fetch data from the previous steps and to store data for the next
        # steps in the database.
        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href
            + "?callbackUrl="
            + url_for(
                f"{INTERACTION_DEMO_BLP.name}.Callback",
                db_id=str(db_task.id),
                _external=True,
            ),
            prog_value=33,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@INTERACTION_DEMO_BLP.route("/<int:db_id>/callback/")
class Callback(MethodView):
    @INTERACTION_DEMO_BLP.arguments(CallbackSchema(unknown=EXCLUDE))
    @INTERACTION_DEMO_BLP.response(HTTPStatus.OK)
    def post(self, arguments, db_id: int):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()
        db_task.data["obj_func_db_id"] = arguments["db_id"]
        db_task.data["number_of_parameters"] = arguments["number_of_parameters"]
        db_task.save(commit=True)

        step_id = "processing-returned-data"
        href = url_for(
            f"{INTERACTION_DEMO_BLP.name}.ProcessStep2View",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{INTERACTION_DEMO_BLP.name}.MicroFrontendStep2",
            db_id=db_task.id,
            _external=True,
        )

        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=66,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()


@INTERACTION_DEMO_BLP.route("/<int:db_id>/ui-step-2/")
class MicroFrontendStep2(MethodView):
    """Micro frontend for the selection of the dataset."""

    example_inputs = {}

    @INTERACTION_DEMO_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the dataset.",
    )
    @INTERACTION_DEMO_BLP.arguments(
        DatasetInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @INTERACTION_DEMO_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the dataset.",
    )
    @INTERACTION_DEMO_BLP.arguments(
        DatasetInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
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

        if not data:
            # retrieve data from the invoked plugin
            obj_func_task_id = db_task.data.get("objective_function_task_id")
            data = {"inputStr": "ID of objective function task: " + str(obj_func_task_id)}

        schema = DatasetInputSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=OptimizerDemo.instance.name,
                version=OptimizerDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.ProcessStep2View",
                    db_id=db_id,  # URL of the second processing step
                ),
                example_values=url_for(
                    f"{INTERACTION_DEMO_BLP.name}.MicroFrontendStep2",  # URL of the second micro frontend
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@INTERACTION_DEMO_BLP.route("/<int:db_id>/process-step-2/")
class ProcessStep2View(MethodView):
    """Start the processing task of step 2."""

    @INTERACTION_DEMO_BLP.arguments(DatasetInputSchema(unknown=EXCLUDE), location="form")
    @INTERACTION_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @INTERACTION_DEMO_BLP.require_jwt("jwt", optional=True)
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

        # Chain the second processing task with executing the task that saves the results and ends the execution.
        task: chain = processing_task_2.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
