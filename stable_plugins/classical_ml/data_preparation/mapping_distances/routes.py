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
from typing import Mapping

from celery.canvas import chain
from flask import Response, redirect, render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from marshmallow import EXCLUDE

from .tasks import calculation_task
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

from . import MAPPING_DISTANCES_BLP, MappingDistances
from .schemas import InputParametersSchema


@MAPPING_DISTANCES_BLP.route("/")
class PluginRootView(MethodView):
    """Plugins collection resource."""

    @MAPPING_DISTANCES_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @MAPPING_DISTANCES_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata"""

        return PluginMetadata(
            title="Taxanomy mapping to distances",
            description=MappingDistances.instance.description,
            name=MappingDistances.instance.name,
            version=MappingDistances.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{MAPPING_DISTANCES_BLP.name}.{ProcessView.__name__}"),
                ui_href=url_for(f"{MAPPING_DISTANCES_BLP.name}.{MicroFrontend.__name__}"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/list",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesUrl",
                    ),
                    InputDataMetadata(
                        data_type="entity/attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesMetadataUrl",
                    ),
                    InputDataMetadata(
                        data_type="graph/taxonomy",
                        content_type=["application/zip"],
                        required=True,
                        parameter="taxonomiesZipUrl",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="relation/element-distances",
                        content_type=["application/zip"],
                        required=True,
                    )
                ],
            ),
            tags=MappingDistances.instance.tags,
        )


@MAPPING_DISTANCES_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Mapping to Distances plugin."""

    @MAPPING_DISTANCES_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the Mapping to Distances plugin.",
    )
    @MAPPING_DISTANCES_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""

        return self.render({}, {}, True)

    @MAPPING_DISTANCES_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the Mapping to Distances plugin.",
    )
    @MAPPING_DISTANCES_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MAPPING_DISTANCES_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with pre-rendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=MappingDistances.instance.name,
                version=MappingDistances.instance.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{MAPPING_DISTANCES_BLP.name}.{ProcessView.__name__}"),
            )
        )


@MAPPING_DISTANCES_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @MAPPING_DISTANCES_BLP.arguments(
        InputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @MAPPING_DISTANCES_BLP.response(HTTPStatus.SEE_OTHER)
    @MAPPING_DISTANCES_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=InputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        task: chain = calculation_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
