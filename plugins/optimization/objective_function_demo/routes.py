from http import HTTPStatus
from typing import Mapping, Optional

import requests
import torch
from celery import chain
from celery.utils.log import get_task_logger
from flask import Response, abort, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    InteractionEndpoint,
    ObjFuncCalcInputSchema,
    ObjFuncCalcOutputSchema,
    ObjFuncCalcInput,
    ObjFuncCalcOutput,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from . import OBJ_FUNC_DEMO_BLP, ObjectiveFunctionDemo
from .neural_network import NeuralNetwork
from .schemas import (
    HyperparametersSchema,
    TaskResponseSchema,
    Hyperparameters,
)
from .tasks import setup_task

TASK_LOGGER = get_task_logger(__name__)


@OBJ_FUNC_DEMO_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @OBJ_FUNC_DEMO_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=ObjectiveFunctionDemo.instance.name,
            description=OBJ_FUNC_DEMO_BLP.description,
            name=ObjectiveFunctionDemo.instance.identifier,
            version=ObjectiveFunctionDemo.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{OBJ_FUNC_DEMO_BLP.name}.{Setup.__name__}"),
                ui_href=url_for(f"{OBJ_FUNC_DEMO_BLP.name}.{MicroFrontend.__name__}"),
                interaction_endpoints=[
                    InteractionEndpoint(
                        type="objective-function-calculation",
                        href=url_for(f"{OBJ_FUNC_DEMO_BLP.name}.{Calculation.__name__}"),
                    )
                ],
                plugin_dependencies=[],
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )
                ],
            ),
            tags=["objective-function"],
        )


@OBJ_FUNC_DEMO_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend of the objective function demo plugin."""

    @OBJ_FUNC_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the objective function demo plugin."
    )
    @OBJ_FUNC_DEMO_BLP.arguments(
        HyperparametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @OBJ_FUNC_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the objective function demo plugin."
    )
    @OBJ_FUNC_DEMO_BLP.arguments(
        HyperparametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        plugin = ObjectiveFunctionDemo.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparametersSchema()

        return Response(
            render_template(
                "simple_template.html",
                name=ObjectiveFunctionDemo.instance.name,
                version=ObjectiveFunctionDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OBJ_FUNC_DEMO_BLP.name}.{Setup.__name__}"
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OBJ_FUNC_DEMO_BLP.name}.{MicroFrontend.__name__}"
                ),  # URL of this endpoint
            )
        )


@OBJ_FUNC_DEMO_BLP.route("/setup/")
class Setup(MethodView):
    """Start the setup task."""

    @OBJ_FUNC_DEMO_BLP.arguments(HyperparametersSchema(unknown=EXCLUDE), location="form")
    @OBJ_FUNC_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: Hyperparameters):
        """Start the setup task."""
        schema = HyperparametersSchema()
        db_task = ProcessingTask(
            task_name=setup_task.name, parameters=schema.dumps(arguments)
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = setup_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)),
            HTTPStatus.SEE_OTHER,
        )


@OBJ_FUNC_DEMO_BLP.route("/calc/")
class Calculation(MethodView):
    """Objective function calculation resource."""

    @OBJ_FUNC_DEMO_BLP.arguments(ObjFuncCalcInputSchema())
    @OBJ_FUNC_DEMO_BLP.response(HTTPStatus.OK, ObjFuncCalcOutputSchema())
    @OBJ_FUNC_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: ObjFuncCalcInput):
        """Endpoint calculating the objective value."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=arguments.db_id)

        if db_task is None:
            msg = (
                f"Could not load task data with id {arguments.db_id} to read parameters!"
            )
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        hyperparameter_url: Optional[str] = None

        for task_file in db_task.outputs:
            if task_file.file_type == "objective-function-hyperparameters":
                hyperparameter_url = STORE.get_task_file_url(task_file)

        if hyperparameter_url is None:
            raise ValueError("Hyperparameter file missing")

        data_set = requests.get(arguments.data_set).json()
        schema = HyperparametersSchema()
        hyperparameters: Hyperparameters = schema.load(
            requests.get(hyperparameter_url).json()
        )

        model = NeuralNetwork(
            hyperparameters.number_of_input_values,
            hyperparameters.number_of_neurons,
        )
        model.set_param_list(arguments.parameters)
        loss = model.get_loss(
            torch.tensor(data_set["input"], dtype=torch.float32),
            torch.tensor(data_set["target"], dtype=torch.float32),
        )

        return ObjFuncCalcOutput(loss)
