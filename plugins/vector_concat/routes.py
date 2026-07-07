# Copyright 2026 QHAna plugin runner contributors.
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
from itertools import chain
from json import dumps
from typing import Mapping

from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from flask_smorest import abort
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import VECTOR_CONCAT_BLP, VectorConcatPlugin
from .schemas import VectorConcatSchema
from .tasks import calculation_task


@VECTOR_CONCAT_BLP.route("/")
class PluginsView(MethodView):
    @VECTOR_CONCAT_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @VECTOR_CONCAT_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return PluginMetadata(
            title="Vector concationation plugin",
            description=VectorConcatPlugin.instance.description,
            name=VectorConcatPlugin.instance.name,
            version=VectorConcatPlugin.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{VECTOR_CONCAT_BLP.name}.CalcView"),
                ui_href=url_for(f"{VECTOR_CONCAT_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "application/json",
                            "application/X-lines+json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="urls",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "text/csv",
                            "application/json",
                            "application/X-lines+json",
                        ],
                        required=True,
                    )
                ],
            ),
            tags=VectorConcatPlugin.instance.tags,
        )


@VECTOR_CONCAT_BLP.route("/ui/")
class MicroFrontend(MethodView):
    @VECTOR_CONCAT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the vector concat plugin."
    )
    @VECTOR_CONCAT_BLP.arguments(
        VectorConcatSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="query",
        required=False,
    )
    @VECTOR_CONCAT_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend"""
        return self.render({}, {}, True)

    @VECTOR_CONCAT_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the vector concat plugin.",
    )
    @VECTOR_CONCAT_BLP.arguments(
        VectorConcatSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True),
        location="form",
        required=False,
    )
    @VECTOR_CONCAT_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = VectorConcatPlugin.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        render_errors = dict(errors) if errors else {}
        schema = VectorConcatSchema()

        return Response(
            render_template(
                "vector-concat.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=render_errors,
                process=url_for(f"{VECTOR_CONCAT_BLP.name}.{CalcView.__name__}"),
            )
        )


@VECTOR_CONCAT_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task"""

    @VECTOR_CONCAT_BLP.arguments(VectorConcatSchema(unknown=EXCLUDE), location="form")
    @VECTOR_CONCAT_BLP.response(HTTPStatus.SEE_OTHER)
    @VECTOR_CONCAT_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=dumps(arguments),
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
