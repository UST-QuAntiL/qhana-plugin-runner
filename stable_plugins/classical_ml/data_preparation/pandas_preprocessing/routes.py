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

import os
from http import HTTPStatus
from typing import Mapping

from pathlib import Path

from celery.canvas import chain
from flask import send_file
from flask import Response
from flask import redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import PDPreprocessing_BLP, PDPreprocessing
from .schemas import InputParametersSchema, TaskResponseSchema
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
    InputDataMetadata,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from .tasks import calculation_task


@PDPreprocessing_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PDPreprocessing_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """pandas-preprocessing endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Pandas Preprocessing",
            description=PDPreprocessing.instance.description,
            name=PDPreprocessing.instance.identifier,
            version=PDPreprocessing.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{PDPreprocessing_BLP.name}.ProcessView"),
                ui_href=url_for(f"{PDPreprocessing_BLP.name}.MicroFrontend"),
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="entity",
                        content_type=[
                            "text/csv",
                        ],
                        required=True,
                    ),
                ],
            ),
            tags=PDPreprocessing.instance.tags,
        )


@PDPreprocessing_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the pandas preprocessing plugin."""

    @PDPreprocessing_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the pandas preprocessing plugin."
    )
    @PDPreprocessing_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @PDPreprocessing_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the pandas preprocessing plugin."
    )
    @PDPreprocessing_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()

        data_dict = dict(data)
        fields = schema.fields
        # define default values
        default_values = {}

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "template.html",
                name=PDPreprocessing.instance.name,
                version=PDPreprocessing.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{PDPreprocessing_BLP.name}.ProcessView"),
                frontendjs=url_for(f"{PDPreprocessing_BLP.name}.get_frontend_js"),
            )
        )


@PDPreprocessing_BLP.route("/ui/frontend_js/")
def get_frontend_js():
    return send_file(Path(__file__).parent / "frontend.js", mimetype="text/javascript")


@PDPreprocessing_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @PDPreprocessing_BLP.arguments(
        InputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @PDPreprocessing_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @PDPreprocessing_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=InputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = calculation_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
