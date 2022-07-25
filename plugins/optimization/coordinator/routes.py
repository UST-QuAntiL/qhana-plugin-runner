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
from . import OPTI_COORD_BLP, OptimizationCoordinator
from .schemas import (
    ObjFuncSelectionSchema,
    TaskResponseSchema,
    DatasetInputSchema,
    ObjFuncCallbackSchema,
    OptimSelectionSchema,
    OptimCallbackSchema,
)
from .tasks import no_op_task, start_optimization_task


@OPTI_COORD_BLP.route("/")
class MetadataView(MethodView):
    """Plugin metadata resource."""

    @OPTI_COORD_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=OptimizationCoordinator.instance.name,
            description=OPTI_COORD_BLP.description,
            name=OptimizationCoordinator.instance.identifier,
            version=OptimizationCoordinator.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(
                    f"{OPTI_COORD_BLP.name}.OptimSetupProcess"
                ),  # URL for the first process endpoint
                ui_href=url_for(
                    f"{OPTI_COORD_BLP.name}.OptimSelectionUI"
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


@OPTI_COORD_BLP.route("/optim-selection/")
class OptimSelectionUI(MethodView):
    """Micro frontend for the selection of the optimizer plugin."""

    # FIXME: remove when plugin selection PR has been merged
    example_inputs = {
        "optimizerUrl": "http://localhost:5005/plugins/optimizer-demo%40v0-1-0",
    }

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the optimizer plugin.",
    )
    @OPTI_COORD_BLP.arguments(
        ObjFuncSelectionSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the optimizer plugin.",
    )
    @OPTI_COORD_BLP.arguments(
        ObjFuncSelectionSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = OptimSelectionSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=OptimizationCoordinator.instance.name,
                version=OptimizationCoordinator.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTI_COORD_BLP.name}.{OptimSetupProcess.__name__}"
                ),  # URL of the first processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OPTI_COORD_BLP.name}.{OptimSelectionUI.__name__}",  # URL of this endpoint
                    **self.example_inputs,
                ),
            )
        )


TASK_LOGGER: Logger = get_task_logger(__name__)


@OPTI_COORD_BLP.route("/optim-setup/")
class OptimSetupProcess(MethodView):
    """
    Adds the setup of the optimizer plugin as the next step. This leads to the micro frontend of the optimizer plugin
    to be displayed to the user who can then input the required hyperparameters.
    """

    @OPTI_COORD_BLP.arguments(OptimSelectionSchema(unknown=EXCLUDE), location="form")
    @OPTI_COORD_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the optimizer setup task."""
        db_task = ProcessingTask(
            task_name=no_op_task.name,
        )
        db_task.save(commit=True)

        optimizer_url = arguments["optimizer_url"]
        db_task.data["optimizer_url"] = optimizer_url
        db_task.save(commit=True)

        # add new step where the setup of the optimizer plugin is executed

        # name of the next step
        step_id = "optimizer-setup"

        # get metadata
        schema = PluginMetadataSchema()
        raw_metadata = requests.get(optimizer_url).json()
        plugin_metadata: PluginMetadata = schema.load(raw_metadata)

        # URL of the process endpoint of the objective function plugin
        href = urljoin(optimizer_url, plugin_metadata.entry_point.href)
        # URL of the micro frontend endpoint of the objective function plugin
        ui_href = urljoin(optimizer_url, plugin_metadata.entry_point.ui_href)

        callback_url_query = "?callbackUrl=" + url_for(
            f"{OPTI_COORD_BLP.name}.{OptimCallback.__name__}",
            db_id=str(db_task.id),
            _external=True,
        )

        # Chain a processing task (that does nothing) with executing the optimization plugin setup.
        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            # adds callback URL so that the optimization plugin can signal that it has finished and send data back
            ui_href=ui_href + callback_url_query,
            prog_value=20,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTI_COORD_BLP.route("/<int:db_id>/optim-callback/")
class OptimCallback(MethodView):
    """
    Callback endpoint that will be used by the optimizer plugins to signal that they have finished and to send
    data back.
    """

    @OPTI_COORD_BLP.arguments(OptimCallbackSchema(unknown=EXCLUDE))
    @OPTI_COORD_BLP.response(HTTPStatus.OK)
    def post(self, arguments, db_id: int):
        """
        Gets data from the objective function plugin and starts the next step.

        :param arguments:
        :param db_id: Database ID of the task object. This refers to the database that the objective function plugin uses which might not be the same as the one this plugin uses.
        :return:
        """
        TASK_LOGGER.info("Optimizer setup callback called")

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()
        db_task.data["optim_db_id"] = arguments["db_id"]
        db_task.save(commit=True)

        step_id = "obj-func-selection"
        href = url_for(
            f"{OPTI_COORD_BLP.name}.{ObjFuncSetupProcess.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{OPTI_COORD_BLP.name}.{ObjFuncSelectionUI.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=40,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()


@OPTI_COORD_BLP.route("/<int:db_id>/obj-func-selection/")
class ObjFuncSelectionUI(MethodView):
    """Micro frontend for the selection of the objective function plugin."""

    # FIXME: remove when plugin selection PR has been merged
    example_inputs = {
        "objectiveFunctionUrl": "http://localhost:5005/plugins/objective-function-demo%40v0-1-0",
    }

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the objective function plugin.",
    )
    @OPTI_COORD_BLP.arguments(
        ObjFuncSelectionSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, errors, db_id)

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the objective function plugin.",
    )
    @OPTI_COORD_BLP.arguments(
        ObjFuncSelectionSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, db_id)

    def render(self, data: Mapping, errors: dict, db_id: int):
        schema = ObjFuncSelectionSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=OptimizationCoordinator.instance.name,
                version=OptimizationCoordinator.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTI_COORD_BLP.name}.{ObjFuncSetupProcess.__name__}", db_id=db_id
                ),  # URL of the first processing step
                help_text="This is an example help text with basic **Markdown** support.",
                example_values=url_for(
                    f"{OPTI_COORD_BLP.name}.{ObjFuncSelectionUI.__name__}",  # URL of this endpoint,
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@OPTI_COORD_BLP.route("/<int:db_id>/obj-func-setup/")
class ObjFuncSetupProcess(MethodView):
    """
    Adds the setup of the objective function plugin as the next step. This leads to the micro frontend of the objective
    function plugin to be displayed to the user who can then input the required hyperparameters.
    """

    @OPTI_COORD_BLP.arguments(ObjFuncSelectionSchema(unknown=EXCLUDE), location="form")
    @OPTI_COORD_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the objective function setup task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()
        objective_function_url = arguments["objective_function_url"]
        db_task.data["objective_function_url"] = objective_function_url
        db_task.save(commit=True)

        # add new step where the setup of the objective function plugin is executed

        # name of the next step
        step_id = "objective-function-setup"

        # get metadata
        schema = PluginMetadataSchema()
        raw_metadata = requests.get(objective_function_url).json()
        plugin_metadata: PluginMetadata = schema.load(raw_metadata)

        # URL of the process endpoint of the objective function plugin
        href = urljoin(objective_function_url, plugin_metadata.entry_point.href)
        # URL of the micro frontend endpoint of the objective function plugin
        ui_href = urljoin(objective_function_url, plugin_metadata.entry_point.ui_href)

        callback_url_query = "?callbackUrl=" + url_for(
            f"{OPTI_COORD_BLP.name}.{ObjFuncCallback.__name__}",
            db_id=str(db_task.id),
            _external=True,
        )

        # Chain a processing task (that does nothing) with executing the objective function plugin setup.
        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            # adds callback URL so that the objective function plugin can signal that it has finished and send data back
            ui_href=ui_href + callback_url_query,
            prog_value=60,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@OPTI_COORD_BLP.route("/<int:db_id>/obj-func-callback/")
class ObjFuncCallback(MethodView):
    """
    Callback endpoint that will be used by the objective function plugins to signal that they have finished and to send
    data back.
    """

    @OPTI_COORD_BLP.arguments(ObjFuncCallbackSchema(unknown=EXCLUDE))
    @OPTI_COORD_BLP.response(HTTPStatus.OK)
    def post(self, arguments, db_id: int):
        """
        Gets data from the objective function plugin and starts the next step.

        :param arguments:
        :param db_id: Database ID of the task object. This refers to the database that the objective function plugin uses which might not be the same as the one this plugin uses.
        :return:
        """
        TASK_LOGGER.info("Objective function setup callback called")

        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()
        db_task.data["obj_func_db_id"] = arguments["db_id"]
        db_task.data["number_of_parameters"] = arguments["number_of_parameters"]
        db_task.save(commit=True)

        step_id = "dataset-selection"
        href = url_for(
            f"{OPTI_COORD_BLP.name}.{StartOptimization.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        ui_href = url_for(
            f"{OPTI_COORD_BLP.name}.{DatasetSelectionUI.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        task: chain = no_op_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
            prog_value=80,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()


@OPTI_COORD_BLP.route("/<int:db_id>/dataset-selection/")
class DatasetSelectionUI(MethodView):
    """Micro frontend for the selection of the dataset."""

    example_inputs = {}

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the dataset.",
    )
    @OPTI_COORD_BLP.arguments(
        DatasetInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @OPTI_COORD_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the selection of the dataset.",
    )
    @OPTI_COORD_BLP.arguments(
        DatasetInputSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
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
                name=OptimizationCoordinator.instance.name,
                version=OptimizationCoordinator.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{OPTI_COORD_BLP.name}.{StartOptimization.__name__}",
                    db_id=db_id,  # URL of the second processing step
                ),
                example_values=url_for(
                    f"{OPTI_COORD_BLP.name}.{DatasetSelectionUI.__name__}",  # URL of the second micro frontend
                    db_id=db_id,
                    **self.example_inputs,
                ),
            )
        )


@OPTI_COORD_BLP.route("/<int:db_id>/optimize/")
class StartOptimization(MethodView):
    """
    Starts the optimization task.
    """

    @OPTI_COORD_BLP.arguments(DatasetInputSchema(unknown=EXCLUDE), location="form")
    @OPTI_COORD_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @OPTI_COORD_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the optimization task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.clear_previous_step()
        db_task.data["dataset_url"] = arguments["dataset_url"]
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = start_optimization_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
