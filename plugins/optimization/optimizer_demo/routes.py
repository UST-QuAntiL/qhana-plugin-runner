# Copyright 2022 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from http import HTTPStatus
from typing import Mapping, Optional, List

import numpy as np
import requests
from celery import chain
from celery.utils.log import get_task_logger
from flask import Response, abort, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
from scipy.optimize import minimize

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    InteractionEndpoint,
    ObjFuncCalcInput,
    ObjFuncCalcInputSchema,
    ObjFuncCalcOutputSchema,
    ObjFuncCalcOutput,
    OptimizationInputSchema,
    OptimizationInput,
    OptimizationOutput,
    OptimizationOutputSchema,
    CallbackURLSchema,
    CallbackURL,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result, add_step
from . import OPTIMIZER_DEMO_BLP, OptimizerDemo
from .schemas import (
    HyperparametersSchema,
    TaskResponseSchema,
    InternalDataSchema,
    Hyperparameters,
    InternalData,
)
from .tasks import setup_task, callback_task, optimization_task

TASK_LOGGER = get_task_logger(__name__)


@OPTIMIZER_DEMO_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @OPTIMIZER_DEMO_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=OptimizerDemo.instance.name,
            description=OPTIMIZER_DEMO_BLP.description,
            name=OptimizerDemo.instance.identifier,
            version=OptimizerDemo.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{OPTIMIZER_DEMO_BLP.name}.{Setup.__name__}"),
                ui_href=url_for(f"{OPTIMIZER_DEMO_BLP.name}.{MicroFrontend.__name__}"),
                interaction_endpoints=[],
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
            tags=["optimizer"],
        )


@OPTIMIZER_DEMO_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend of the optimizer plugin."""

    @OPTIMIZER_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the optimizer plugin."
    )
    @OPTIMIZER_DEMO_BLP.arguments(
        HyperparametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTIMIZER_DEMO_BLP.arguments(
        CallbackURLSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, callback_url: CallbackURL):
        """Return the micro frontend."""
        return self.render(request.args, errors, callback_url)

    @OPTIMIZER_DEMO_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the optimizer plugin."
    )
    @OPTIMIZER_DEMO_BLP.arguments(
        HyperparametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTIMIZER_DEMO_BLP.arguments(
        CallbackURLSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, callback_url: CallbackURL):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, callback_url)

    def render(self, data: Mapping, errors: dict, callback_url: CallbackURL):
        plugin = OptimizerDemo.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparametersSchema()

        return Response(
            render_template(
                "simple_template.html",
                name=OptimizerDemo.instance.name,
                version=OptimizerDemo.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTIMIZER_DEMO_BLP.name}.{Setup.__name__}",
                    **CallbackURLSchema().dump(callback_url),
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OPTIMIZER_DEMO_BLP.name}.{MicroFrontend.__name__}",
                    **CallbackURLSchema().dump(callback_url),
                ),  # URL of this endpoint
            )
        )


@OPTIMIZER_DEMO_BLP.route("/setup/")
class Setup(MethodView):
    """Start the setup task."""

    @OPTIMIZER_DEMO_BLP.arguments(HyperparametersSchema(unknown=EXCLUDE), location="form")
    @OPTIMIZER_DEMO_BLP.arguments(
        CallbackURLSchema(unknown=EXCLUDE),
        location="query",
        required=False,
    )
    @OPTIMIZER_DEMO_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: Hyperparameters, callback_url: CallbackURL):
        """Start the setup task."""
        schema = InternalDataSchema()
        internal_data = InternalData(hyperparameters=arguments, callback_url=callback_url)

        db_task = ProcessingTask(
            task_name=setup_task.name, parameters=schema.dumps(internal_data)
        )
        db_task.save(commit=True)

        optimizer_start_url = url_for(
            f"{OPTIMIZER_DEMO_BLP.name}.{Optimization.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        task_url = url_for("tasks-api.TaskView", task_id=str(db_task.id), _external=True)

        # all tasks need to know about db id to load the db entry
        task: chain = (
            setup_task.s(db_id=db_task.id)
            | add_step.s(
                db_id=db_task.id,
                step_id="optimization",
                href=optimizer_start_url,
                ui_href="",
                prog_value=50,
                forget_result=False,
            )
            | callback_task.s(
                callback_url=callback_url.callback_url,
                task_url=task_url,
            )
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)),
            HTTPStatus.SEE_OTHER,
        )


@OPTIMIZER_DEMO_BLP.route("/<int:db_id>/optimization/")
class Optimization(MethodView):
    """Start the optimization."""

    @OPTIMIZER_DEMO_BLP.arguments(OptimizationInputSchema(unknown=EXCLUDE))
    @OPTIMIZER_DEMO_BLP.response(HTTPStatus.OK, OptimizationOutputSchema())
    @OPTIMIZER_DEMO_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: OptimizationInput, db_id: int):
        """Start the optimization."""

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()
        db_task.save(commit=True)

        schema = InternalDataSchema()
        parameters: InternalData = schema.loads(db_task.parameters)

        parameters.optimization_input = arguments

        db_task.parameters = schema.dumps(parameters)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = optimization_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )