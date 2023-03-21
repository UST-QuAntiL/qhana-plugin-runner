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

from celery.canvas import chain
from flask import Response
from flask import redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import MaxCut_BLP, MaxCut
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


@MaxCut_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @MaxCut_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @MaxCut_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """max_cut endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Max Cut",
            description=MaxCut.instance.description,
            name=MaxCut.instance.identifier,
            version=MaxCut.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{MaxCut_BLP.name}.ProcessView"),
                ui_href=url_for(f"{MaxCut_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                        parameter="similarityMatrix",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/label",
                        content_type=[
                            "application/json",
                        ],
                        required=True,
                    ),
                ],
            ),
            tags=MaxCut.instance.tags,
        )


@MaxCut_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the max cut plugin."""

    @MaxCut_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the max cut plugin."
    )
    @MaxCut_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MaxCut_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @MaxCut_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the max cut plugin."
    )
    @MaxCut_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MaxCut_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()

        data_dict = dict(data)
        fields = schema.fields
        # define default values
        default_values = {
            fields["num_clusters"].data_key: 1,
            fields["max_trials"].data_key: 10,
            fields["reps"].data_key: 4,
            fields["shots"].data_key: 1000,
        }

        if "IBMQ_TOKEN" in os.environ:
            default_values[fields["ibmq_token"].data_key] = "****"

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=MaxCut.instance.name,
                version=MaxCut.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{MaxCut_BLP.name}.ProcessView"),
            )
        )


@MaxCut_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @MaxCut_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @MaxCut_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MaxCut_BLP.require_jwt("jwt", optional=True)
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
