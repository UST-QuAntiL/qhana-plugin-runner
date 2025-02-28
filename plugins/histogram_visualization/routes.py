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

import hashlib
from http import HTTPStatus
from typing import Mapping
from celery import chain
import celery
from flask import abort, redirect, send_file
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask.wrappers import Response
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState

from . import VIS_BLP, HistogramVisualization
from .schemas import HistogramInputParametersSchema
from .tasks import generate_plot, process


@VIS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @VIS_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @VIS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = HistogramVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.visualization,
            entry_point=EntryPoint(
                href=url_for(f"{VIS_BLP.name}.ProcessView"),
                ui_href=url_for(f"{VIS_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                        parameter="data",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="image/html",
                        content_type=["text/html"],
                        required=True,
                    )
                ],
            ),
            tags=["visualization", "histogram"],
        )


@VIS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the histogram visualization plugin."""

    @VIS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the histogram visualization plugin.",
    )
    @VIS_BLP.arguments(
        HistogramInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @VIS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @VIS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the histogram visualization plugin.",
    )
    @VIS_BLP.arguments(
        HistogramInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @VIS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = HistogramVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return Response(
            render_template(
                "histogram_visualization.html",
                name=plugin.name,
                version=plugin.version,
                schema=HistogramInputParametersSchema(),
                valid=valid,
                values=data,
                errors=errors,
                example_values=url_for(f"{VIS_BLP.name}.MicroFrontend"),
                get_plot_url=url_for(f"{VIS_BLP.name}.get_plot"),
                process=url_for(f"{VIS_BLP.name}.ProcessView"),
            )
        )


@VIS_BLP.route("/plots/")
@VIS_BLP.response(HTTPStatus.OK, description="Histogram Visualization.")
@VIS_BLP.arguments(
    HistogramInputParametersSchema(unknown=EXCLUDE),
    location="query",
    required=True,
)
@VIS_BLP.require_jwt("jwt", optional=True)
# Method called through the micro frontend, when a data_url is selected
def get_plot(data: Mapping):
    data_url = data.get("data", None)

    # Check that data url is present
    if not data_url:
        abort(HTTPStatus.BAD_REQUEST)
    url_hash = hashlib.sha256(data_url.encode("utf-8")).hexdigest()
    plot = DataBlob.get_value(HistogramVisualization.instance.identifier, url_hash, None)
    if plot is None:
        if not (
            task_id := PluginState.get_value(
                HistogramVisualization.instance.identifier, url_hash, None
            )
        ):
            # Add the generate_table from task.py as an async method
            task_result = generate_plot.s(data_url, url_hash).apply_async()
            PluginState.set_value(
                HistogramVisualization.instance.identifier,
                url_hash,
                task_result.id,
                commit=True,
            )
        else:
            task_result = CELERY.AsyncResult(task_id)
        try:
            task_result.get(timeout=5)
            # Retrieve the generated HTML
            plot = DataBlob.get_value(
                HistogramVisualization.instance.identifier, url_hash
            )
        except celery.exceptions.TimeoutError:
            return Response("Plot not yet created!", HTTPStatus.ACCEPTED)
    if not plot:
        abort(HTTPStatus.BAD_REQUEST, "Invalid data URL!")
    # Return Html to the micro frontend
    return Response(plot)


@VIS_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @VIS_BLP.arguments(HistogramInputParametersSchema(unknown=EXCLUDE), location="form")
    @VIS_BLP.response(HTTPStatus.SEE_OTHER)
    @VIS_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        data = arguments.get("data", None)
        if data is None:
            abort(HTTPStatus.BAD_REQUEST)
        url_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
        db_task = ProcessingTask(task_name=process.name)
        db_task.save(commit=True)
        # all tasks need to know about db id to load the db entry
        task: chain = process.s(
            db_id=db_task.id, data_url=data, hash=url_hash
        ) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async(db_id=db_task.id, data_url=data, hash=url_hash)

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
