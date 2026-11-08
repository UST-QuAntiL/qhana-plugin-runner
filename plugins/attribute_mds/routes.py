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
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
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

from . import ATTRIBUTE_MDS_BLP, AttributeMds
from .schemas import InputParametersSchema, MetricEnum, MissingDataHandling
from .tasks import calculation_task


@ATTRIBUTE_MDS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @ATTRIBUTE_MDS_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @ATTRIBUTE_MDS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Attribute distance MDS endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Multidimensional Scaling (MDS) on Attribute Distances",
            description=AttributeMds.instance.description,
            name=AttributeMds.instance.name,
            version=AttributeMds.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{ATTRIBUTE_MDS_BLP.name}.CalcView"),
                ui_href=url_for(f"{ATTRIBUTE_MDS_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="relation/attribute-distances",
                        content_type=["application/zip"],
                        required=True,
                        parameter="attributeDistancesUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=["application/zip"],
                        required=True,
                    )
                ],
            ),
            tags=AttributeMds.instance.tags,
        )


@ATTRIBUTE_MDS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the attribute distance MDS plugin."""

    @ATTRIBUTE_MDS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the attribute distance MDS plugin.",
    )
    @ATTRIBUTE_MDS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @ATTRIBUTE_MDS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @ATTRIBUTE_MDS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the attribute distance MDS plugin.",
    )
    @ATTRIBUTE_MDS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @ATTRIBUTE_MDS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        fields = InputParametersSchema().fields

        default_values = {
            fields["dimensions"].data_key: 2,
            fields["metric"].data_key: MetricEnum.metric_mds.name,
            fields["n_init"].data_key: 4,
            fields["max_iter"].data_key: 300,
            fields["missing_data_handling"].data_key: MissingDataHandling.mean.name,
        }
        default_values.update(data)

        return Response(
            render_template(
                "simple_template.html",
                name=AttributeMds.instance.name,
                version=AttributeMds.instance.version,
                schema=InputParametersSchema(),
                valid=valid,
                values=default_values,
                errors=errors,
                process=url_for(f"{ATTRIBUTE_MDS_BLP.name}.CalcView"),
            )
        )


@ATTRIBUTE_MDS_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @ATTRIBUTE_MDS_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @ATTRIBUTE_MDS_BLP.response(HTTPStatus.SEE_OTHER)
    @ATTRIBUTE_MDS_BLP.require_jwt("jwt", optional=True)
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
