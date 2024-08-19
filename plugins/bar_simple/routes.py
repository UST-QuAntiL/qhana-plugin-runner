# Copyright 2023 QHAna plugin runner contributors.
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
from json import dumps
from typing import Mapping

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect, abort
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
    InputDataMetadata,
    DataMetadata
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import (
    TASK_STEPS_CHANGED,
    add_step,
    save_task_error,
    save_task_result,
)

from . import BAR_BLP, BarDiagram
from .schemas import InputParametersSchema, TaskResponseSchema
from .tasks import visualization_task

@BAR_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @BAR_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @BAR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        
        return PluginMetadata(
            title="Bar Diagram",
            description=BarDiagram.instance.description,
            name=BarDiagram.instance.name,
            version=BarDiagram.instance.version,
            type=PluginType.visualization,
            entry_point=EntryPoint(
                href=url_for(f"{BAR_BLP.name}.ProcessView"),
                ui_href=url_for(f"{BAR_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/label",
                        content_type=["application/json"],
                        required=True,
                        parameter="clustersUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="plot",
                        content_type=["text/html"],
                        required=True
                    )
                ],
            ),
            tags=BarDiagram.instance.tags,
        )


@BAR_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Simple Bar Diagram plugin."""

    @BAR_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the Simple Bar Diagram plugin."
    )
    @BAR_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @BAR_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @BAR_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for the Simple Bar Diagram plugin."
    )
    @BAR_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @BAR_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        
        data_dict = dict(data)

        # define default values
        default_values = {}

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values
        
        return Response(
            render_template(
                "simple_template.html",
                name=BarDiagram.instance.name,
                version=BarDiagram.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{BAR_BLP.name}.ProcessView"),
            )
        )


@BAR_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @BAR_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @BAR_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @BAR_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the visualization task."""
        db_task = ProcessingTask(
            task_name=visualization_task.name, 
            parameters=InputParametersSchema().dumps(arguments)
            )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = visualization_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
