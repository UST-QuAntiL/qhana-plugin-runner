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

import os
from http import HTTPStatus
from typing import Mapping, Optional
from logging import Logger

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response
from flask import redirect, abort
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import RIDGELOSS_BLP, RidgeLoss
from .schemas import HyperparamterInputSchema, RidgeLossTaskResponseSchema
from .tasks import optimize
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result, add_step

TASK_LOGGER = get_task_logger(__name__)


@RIDGELOSS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins metadata resource."""

    @RIDGELOSS_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        return PluginMetadata(
            title=RidgeLoss.instance.name,
            description=RidgeLoss.instance.description,
            name=RidgeLoss.instance.identifier,
            version=RidgeLoss.instance.version,
            type=PluginType.processing,
            tags=RidgeLoss.instance.tags,
            entry_point=EntryPoint(
                href=url_for(
                    f"{RIDGELOSS_BLP.name}.{OptimizationProcessView.__name__}", db_id=0
                ),  # URL for the first process endpoint  # FIXME: db_id
                ui_href=url_for(
                    f"{RIDGELOSS_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
                    db_id=0,
                ),  # URL for the first micro frontend endpoint  # FIXME: db_id
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
        )


@RIDGELOSS_BLP.route("/<int:db_id>/ui-hyperparameter/")
class HyperparameterSelectionMicroFrontend(MethodView):
    """Micro frontend for the hyperparameter selection."""

    example_inputs = {
        "alpha": 0.1,
    }

    @RIDGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @RIDGELOSS_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors)

    @RIDGELOSS_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the hyperparameter selection."
    )
    @RIDGELOSS_BLP.arguments(
        HyperparamterInputSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors)

    def render(self, data: Mapping, db_id: int, errors: dict):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        plugin = RidgeLoss.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        schema = HyperparamterInputSchema()

        if not data:
            data = {"alpha": 0.1}

        return Response(
            render_template(
                "simple_template.html",
                name=RidgeLoss.instance.name,
                version=RidgeLoss.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{RIDGELOSS_BLP.name}.{OptimizationProcessView.__name__}",
                    db_id=db_id,
                ),  # URL of the processing step
                help_text="This is an example help text with basic **Markdown** support.",
                # TODO: give a proper description what alpha is with a link to the documentation
                example_values=url_for(
                    f"{RIDGELOSS_BLP.name}.{HyperparameterSelectionMicroFrontend.__name__}",
                    db_id=db_id,
                    **self.example_inputs,
                ),  # URL of this endpoint
            )
        )


@RIDGELOSS_BLP.route("/<int:db_id>/process-optimizer/")
class OptimizationProcessView(MethodView):
    """Start a long running processing task."""

    @RIDGELOSS_BLP.arguments(HyperparamterInputSchema(unknown=EXCLUDE), location="form")
    @RIDGELOSS_BLP.response(HTTPStatus.OK, RidgeLossTaskResponseSchema())
    @RIDGELOSS_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        """Start the invoked task."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.data["alpha"] = arguments["alpha"]

        href = db_task.data["opt_href"]

        ui_href = db_task.data["opt_ui_href"]
        db_task.clear_previous_step()
        db_task.save(commit=True)
        # add the next processing step with the data that was stored by the previous step of the invoking plugin
        task: chain = optimize.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id,
            step_id="optimizer callback",  # name of the next sub-step
            href=href,  # URL to the processing endpoint of the next step
            ui_href=ui_href,  # URL to the micro frontend of the next step
            prog_value=90,
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        # start tasks
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )