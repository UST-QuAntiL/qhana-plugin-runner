# Copyright 2024 QHAna plugin runner contributors.
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

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import M4MLoader_BLP, M4MLoaderPlugin
from .schemas import InputParametersSchema
from .tasks import import_data

TASK_LOGGER = get_task_logger(__name__)


@M4MLoader_BLP.route("/")
class PluginsView(MethodView):
    """Plugin root resource."""

    @M4MLoader_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @M4MLoader_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Main endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="MUSE4Music Loader",
            description=M4MLoaderPlugin.instance.description,
            name=M4MLoaderPlugin.instance.name,
            version=M4MLoaderPlugin.instance.version,
            type=PluginType.dataloader,
            entry_point=EntryPoint(
                href=url_for(f"{M4MLoader_BLP.name}.{ProcessView.__name__}"),
                ui_href=url_for(f"{M4MLoader_BLP.name}.{MicroFrontend.__name__}"),
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="entity/list",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="entity/attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="graph/taxonomy",
                        content_type=["application/zip"],
                        required=True,
                    ),
                ],
            ),
            tags=M4MLoaderPlugin.instance.tags,
        )


@M4MLoader_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the MUSE4Music loader plugin."""

    @M4MLoader_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the MUSE4Music loader plugin."
    )
    @M4MLoader_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @M4MLoader_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @M4MLoader_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the MUSE4Music loader plugin."
    )
    @M4MLoader_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @M4MLoader_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()

        data_dict = dict(data)
        # define default values
        default_values = {
            "url": "",
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=M4MLoaderPlugin.instance.name,
                version=M4MLoaderPlugin.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{M4MLoader_BLP.name}.{ProcessView.__name__}"),
            )
        )


@M4MLoader_BLP.route("/process/")
class ProcessView(MethodView):
    """Start importing MUSE4Music data."""

    @M4MLoader_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @M4MLoader_BLP.response(HTTPStatus.FOUND)
    @M4MLoader_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the data import task."""
        db_task = ProcessingTask(
            task_name=import_data.name,
            parameters=InputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = import_data.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
