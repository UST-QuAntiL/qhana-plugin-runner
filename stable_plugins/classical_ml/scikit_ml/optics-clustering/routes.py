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

from . import Optics_BLP, Optics
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


@Optics_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @Optics_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @Optics_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """optics endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Optics",
            description=Optics.instance.description,
            name=Optics.instance.name,
            version=Optics.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{Optics_BLP.name}.ProcessView"),
                ui_href=url_for(f"{Optics_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                        parameter="entityPointsUrl",
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
                    DataMetadata(
                        data_type="plot",
                        content_type=["text/html"],
                        required=False,
                    ),
                ],
            ),
            tags=Optics.instance.tags,
        )


@Optics_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the optics plugin."""

    @Optics_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the optics plugin."
    )
    @Optics_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @Optics_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @Optics_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the optics plugin."
    )
    @Optics_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @Optics_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()

        data_dict = dict(data)
        fields = schema.fields
        # define default values
        default_values = {
            fields["min_samples"].data_key: 5,
            fields["max_epsilon"].data_key: -1,
            fields["minkowski_p"].data_key: 2,
            fields["epsilon"].data_key: -1,
            fields["xi"].data_key: 0.05,
            fields["min_cluster_size"].data_key: -1,
            fields["leaf_size"].data_key: 30,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "optics-clustering_template.html",
                name=Optics.instance.name,
                version=Optics.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{Optics_BLP.name}.ProcessView"),
            )
        )


@Optics_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @Optics_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @Optics_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @Optics_BLP.require_jwt("jwt", optional=True)
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
