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

from celery import chain
from flask import send_file
from flask import url_for, request, Response, render_template, redirect
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import HybridAutoencoderPlugin, HA_BLP
from .tasks import hybrid_autoencoder_pennylane_task
from .schemas import (
    HybridAutoencoderPennylaneRequestSchema,
)
from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
    InputDataMetadata,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_result, save_task_error


@HA_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @HA_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @HA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Demo endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Hybrid Autoencoder",
            description=HybridAutoencoderPlugin.instance.description,
            name=HybridAutoencoderPlugin.instance.name,
            version=HybridAutoencoderPlugin.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{HA_BLP.name}.HybridAutoencoderPennylaneAPI"),
                ui_href=url_for(f"{HA_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="trainPointsUrl",
                    ),
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=False,
                        parameter="testPointsUrl",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="qnn-weights",
                        content_type=["application/json"],
                        required=True,
                    ),
                ],
            ),
            tags=HybridAutoencoderPlugin.instance.tags,
        )


@HA_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the hybrid autoencoder plugin."""

    example_inputs = {
        "inputData": "data:text/plain,0,0,0,0,0,0,0,0,0,0",
        "numberOfQubits": 3,
        "embeddingSize": 2,
        "qnnName": "QNN3",
        "trainingSteps": 100,
    }

    @HA_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hybrid autoencoder plugin."
    )
    @HA_BLP.arguments(
        HybridAutoencoderPennylaneRequestSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @HA_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @HA_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the hybrid autoencoder plugin."
    )
    @HA_BLP.arguments(
        HybridAutoencoderPennylaneRequestSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @HA_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        # Render schema errors on fields
        schema_error = errors.get("_schema", None)
        if schema_error:
            if (
                "The number of qubits must be greater or equal to the embedding size."
                in schema_error
            ):
                errors["numberOfQubits"] = errors.get("numberOfQubits", []) + [
                    "The number of qubits must be greater or equal to the embedding size."
                ]
                errors["embeddingSize"] = errors.get("embeddingSize", []) + [
                    "The embedding size must be less or equal to the number of qubits."
                ]
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        schema = HybridAutoencoderPennylaneRequestSchema()

        data_dict = dict(data)
        fields = HybridAutoencoderPennylaneRequestSchema().fields

        # define default values
        default_values = {
            fields["number_of_qubits"].data_key: 3,
            fields["embedding_size"].data_key: 2,
            fields["training_steps"].data_key: 100,
            fields["shots"].data_key: 100,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "hybrid_ae_pkg_template.html",
                name=HybridAutoencoderPlugin.instance.name,
                version=HybridAutoencoderPlugin.instance.version,
                schema=schema,
                valid=valid,
                values=data_dict,
                errors=errors,
                process=url_for(f"{HA_BLP.name}.HybridAutoencoderPennylaneAPI"),
            )
        )


@HA_BLP.route("/process/pennylane/")
class HybridAutoencoderPennylaneAPI(MethodView):
    """Start a long running processing task."""

    @HA_BLP.response(HTTPStatus.SEE_OTHER)
    @HA_BLP.arguments(
        HybridAutoencoderPennylaneRequestSchema(unknown=EXCLUDE), location="form"
    )
    @HA_BLP.require_jwt("jwt", optional=True)
    def post(self, req_dict):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name=hybrid_autoencoder_pennylane_task.name,
            parameters=HybridAutoencoderPennylaneRequestSchema().dumps(req_dict),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = hybrid_autoencoder_pennylane_task.s(
            db_id=db_task.id
        ) | save_task_result.s(db_id=db_task.id)
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
