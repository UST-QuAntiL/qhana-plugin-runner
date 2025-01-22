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

from . import VIS_BLP, ZXCalculusVisualization
from .schemas import ZXCalculusInputParametersSchema
from .tasks import generate_image, process


@VIS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @VIS_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @VIS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata."""
        plugin = ZXCalculusVisualization.instance
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
                        data_type="executable/circuit",
                        content_type=["text/x-qasm"],
                        required=True,
                        parameter="data",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="circuit",
                        content_type=["text/html"],
                        required=True,
                    )
                ],
            ),
            tags=["visualization", "zxcalculus", "circuit"],
        )


@VIS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the ZXCalculus visualization plugin."""

    @VIS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the ZXCalcuclus visualization plugin.",
    )
    @VIS_BLP.arguments(
        ZXCalculusInputParametersSchema(
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
        description="Micro frontend of the ZXCalculus visualization plugin.",
    )
    @VIS_BLP.arguments(
        ZXCalculusInputParametersSchema(
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
        plugin = ZXCalculusVisualization.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        return Response(
            render_template(
                "zxcalculus_visualization.html",
                name=plugin.name,
                version=plugin.version,
                schema=ZXCalculusInputParametersSchema(),
                valid=valid,
                values=data,
                errors=errors,
                example_values=url_for(f"{VIS_BLP.name}.MicroFrontend"),
                get_image_url=url_for(f"{VIS_BLP.name}.get_image"),
                process=url_for(f"{VIS_BLP.name}.ProcessView"),
            )
        )


@VIS_BLP.route("/circuits/")
@VIS_BLP.response(HTTPStatus.OK, description="ZXCalculus Visualization.")
@VIS_BLP.arguments(
    ZXCalculusInputParametersSchema(unknown=EXCLUDE),
    location="query",
    required=True,
)
@VIS_BLP.require_jwt("jwt", optional=True)
# Method called through the micro frontend, when a data_url is selected,
# or when the Optimized Checkbox is changed
def get_image(data: Mapping):
    data_url = data.get("data", None)
    optimized = data.get("optimized", None)

    # Data_Url needs to be provided, while optimized is always provided through the Micro frontend
    if not data_url or optimized == None:
        abort(HTTPStatus.BAD_REQUEST)

    # Two circuits are created with two different hashes
    url_hash_norm = hashlib.sha256(data_url.encode("utf-8")).hexdigest()
    url_hash_opt = hashlib.sha256((data_url + "_optimized").encode("utf-8")).hexdigest()
    image = DataBlob.get_value(
        ZXCalculusVisualization.instance.identifier,
        url_hash_opt if optimized else url_hash_norm,
        None,
    )
    if image is None:
        if not (
            task_id := PluginState.get_value(
                ZXCalculusVisualization.instance.identifier, url_hash_norm, None
            )
        ):
            # Add the generate_image from task.py as an async method 
            task_result = generate_image.s(
                data_url, url_hash_norm, url_hash_opt
            ).apply_async()
            PluginState.set_value(
                ZXCalculusVisualization.instance.identifier,
                url_hash_norm,
                task_result.id,
                commit=True,
            )
        else:
            task_result = CELERY.AsyncResult(task_id)
        try:
            task_result.get(timeout=5)
            # Both images were created, retrieve the correct one
            image = DataBlob.get_value(
                ZXCalculusVisualization.instance.identifier,
                url_hash_opt if optimized else url_hash_norm,
            )
        except celery.exceptions.TimeoutError:
            return Response("Circuit Image not yet created!", HTTPStatus.ACCEPTED)
    if not image:
        abort(HTTPStatus.BAD_REQUEST, "Invalid data URL!")
    # Returns html to the micro frontend
    return Response(image)


@VIS_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @VIS_BLP.arguments(ZXCalculusInputParametersSchema(unknown=EXCLUDE), location="form")
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
