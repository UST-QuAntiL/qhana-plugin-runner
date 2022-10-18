# Copyright 2022 QHAna plugin runner contributors.
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
from flask import Response
from flask import redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import PCA_BLP, PCA
from .schemas import (
    InputParametersSchema,
    TaskResponseSchema,
    SolverEnum,
    PCATypeEnum,
    KernelEnum,
)
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


@PCA_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PCA_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @PCA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """PCA endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Principle Component Analysis (PCA)",
            description=PCA.instance.description,
            name=PCA.instance.identifier,
            version=PCA.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{PCA_BLP.name}.ProcessView"),
                ui_href=url_for(f"{PCA_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=["text/csv", "application/json"],
                        required=True,
                        parameter="entityPointsUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="plot",
                        content_type=["text/html"],
                        required=False,
                    ),
                    DataMetadata(
                        data_type="pca-metadata",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=["text/csv"],
                        required=True,
                    ),
                ],
            ),
            tags=PCA.instance.tags,
        )


@PCA_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the PCA plugin."""

    @PCA_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the PCA plugin.")
    @PCA_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @PCA_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @PCA_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the PCA plugin.")
    @PCA_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @PCA_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        # Render schema errors on fields
        schema_error = errors.get("_schema", None)
        if schema_error:
            if "Entity points url must not be none." in schema_error:
                errors["entityPointsUrl"] = errors.get("entityPointsUrl", []) + ["Field may not be null."]
            elif "Kernel url must not be none." in schema_error:
                errors["kernelUrl"] = errors.get("kernelUrl", []) + ["Field may not be null."]

        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)
        fields = InputParametersSchema().fields

        # define default values
        default_values = {
            fields["pca_type"].data_key: PCATypeEnum.normal,
            fields["dimensions"].data_key: 1,
            fields["solver"].data_key: SolverEnum.auto,
            fields["batch_size"].data_key: 1,
            fields["sparsity_alpha"].data_key: 1,
            fields["ridge_alpha"].data_key: 0.01,
            fields["kernel"].data_key: KernelEnum.linear,
            fields["degree"].data_key: 3,
            fields["kernel_gamma"].data_key: 0.1,
            fields["kernel_coef"].data_key: 1,
            fields["max_itr"].data_key: 1000,
            fields["tol"].data_key: 0,
            fields["iterated_power"].data_key: 0,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "pca_template.html",
                name=PCA.instance.name,
                version=PCA.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{PCA_BLP.name}.ProcessView"),
            )
        )


@PCA_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @PCA_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @PCA_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @PCA_BLP.require_jwt("jwt", optional=True)
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
