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
from celery.exceptions import TimeoutError
from flask import abort, redirect
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
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import TDA_BLP, TDAVisualization
from .schemas import TDAInputParametersSchema
from .tasks import generate_plot, process


@TDA_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @TDA_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @TDA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = TDAVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return PluginMetadata(
            title=plugin.name,
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.visualization,
            entry_point=EntryPoint(
                href=url_for(f"{TDA_BLP.name}.ProcessView"),
                ui_href=url_for(f"{TDA_BLP.name}.MicroFrontend"),
                plugin_dependencies=[],
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json", "application/csv"],
                        required=True,
                        parameter="entityUrl",
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
            tags=plugin.tags,
        )


@TDA_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the tda visualization plugin."""

    @TDA_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the tda visualization plugin.",
    )
    @TDA_BLP.arguments(
        TDAInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @TDA_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @TDA_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the tda visualization plugin.",
    )
    @TDA_BLP.arguments(
        TDAInputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @TDA_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = TDAVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return Response(
            render_template(
                "tda_visualization.html",
                name=plugin.name,
                version=plugin.version,
                schema=TDAInputParametersSchema(),
                valid=valid,
                values=data,
                errors=errors,
                example_values=url_for(f"{TDA_BLP.name}.MicroFrontend"),
                get_plot_url=url_for(f"{TDA_BLP.name}.get_plot"),
                process=url_for(f"{TDA_BLP.name}.ProcessView"),
            )
        )


@TDA_BLP.route("/plots/")
@TDA_BLP.response(HTTPStatus.OK, description="TDA Visualization.")
@TDA_BLP.arguments(
    TDAInputParametersSchema(unknown=EXCLUDE),
    location="query",
    required=True,
)
@TDA_BLP.require_jwt("jwt", optional=True)
# Method called through the micro frontend, when an entity_url is selected,
# or when a cluster_url is selected, whil an entity_url is present TODO
def get_plot(data: Mapping):
    entity_url = data.get("entity_url", None)
    homology_dimension = data.get("homology_dimension", 1)
    # Only an entity_url is required to generate a plot
    if entity_url is None:
        abort(HTTPStatus.BAD_REQUEST)
    # As the homology_dimension can be null, the str method is required
    url_hash = hashlib.sha256(
        (entity_url + str(homology_dimension)).encode("utf-8")
    ).hexdigest()
    plot = DataBlob.get_value(TDAVisualization.instance.identifier, url_hash, None)
    if plot is None:
        if not (
            task_id := PluginState.get_value(
                TDAVisualization.instance.identifier, url_hash, None
            )
        ):
            # Add the generate_plot from task.py as an async method
            task_result = generate_plot.s(
                entity_url, homology_dimension, hash_=url_hash
            ).apply_async()
            PluginState.set_value(
                TDAVisualization.instance.identifier,
                url_hash,
                task_result.id,
                commit=True,
            )
        else:
            task_result = CELERY.AsyncResult(task_id)
        try:
            task_result.get(timeout=5)
            # Retrieve the generated html
            plot = DataBlob.get_value(TDAVisualization.instance.identifier, url_hash)
        except TimeoutError:
            return Response("Plot not yet created!", HTTPStatus.ACCEPTED)
    if not plot:
        abort(HTTPStatus.BAD_REQUEST, "Invalid circuit URL!")

    # Returns the html to the micro frontend
    return Response(plot)


@TDA_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @TDA_BLP.arguments(TDAInputParametersSchema(unknown=EXCLUDE), location="form")
    @TDA_BLP.response(HTTPStatus.SEE_OTHER)
    @TDA_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        entity_url = arguments.get("entity_url", None)
        homology_dimension = arguments.get("homology_dimension", None)
        if entity_url is None:
            abort(HTTPStatus.BAD_REQUEST)
        url_hash = hashlib.sha256(
            (entity_url + str(homology_dimension)).encode("utf-8")
        ).hexdigest()
        db_task = ProcessingTask(task_name=process.name)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = process.s(
            db_id=db_task.id, entity_url=entity_url, homology_dimension=homology_dimension
        ) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
