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

from . import VQC_BLP, VQC
from .backend.qiskit_backends import QiskitBackends
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


@VQC_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @VQC_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @VQC_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Quantum variational classifier endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Quantum Variational Classifier",
            description=VQC.instance.description,
            name=VQC.instance.name,
            version=VQC.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{VQC_BLP.name}.CalcView"),
                ui_href=url_for(f"{VQC_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="train_data_url",
                    ),
                    InputDataMetadata(
                        data_type="entity/label",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="train_labels_url",
                    ),
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="test_data_url",
                    ),
                    InputDataMetadata(
                        data_type="entity/label",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=False,
                        parameter="test_labels_url",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/label",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="plot",
                        content_type=["text/html"],
                        required=False,
                    ),
                    DataMetadata(
                        data_type="plot",
                        content_type=["text/html"],
                        required=False,
                    ),
                    DataMetadata(
                        data_type="representative-circuit",
                        content_type=["application/qasm"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="vqc-metadata",
                        content_type=["application/json"],
                        required=True,
                    ),
                ],
            ),
            tags=VQC.instance.tags,
        )


@VQC_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the quantum variational classifier plugin."""

    @VQC_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the quantum variational classifier plugin.",
    )
    @VQC_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @VQC_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @VQC_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the quantum variational classifier plugin.",
    )
    @VQC_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @VQC_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)
        fields = InputParametersSchema().fields

        # define default values
        default_values = {
            fields["reps_feature_map"].data_key: 2,
            fields["reps_ansatz"].data_key: 2,
            fields["maxitr"].data_key: 1000,
            fields["shots"].data_key: 1024,
            fields["backend"].data_key: QiskitBackends.aer_statevector_simulator.value,
            fields["resolution"].data_key: 20,
        }

        if "IBMQ_BACKEND" in os.environ:
            default_values[fields["backend"].data_key] = os.environ["IBMQ_BACKEND"]

        if "IBMQ_TOKEN" in os.environ:
            default_values[fields["ibmq_token"].data_key] = "****"

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=VQC.instance.name,
                version=VQC.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{VQC_BLP.name}.CalcView"),
            )
        )


@VQC_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @VQC_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @VQC_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @VQC_BLP.require_jwt("jwt", optional=True)
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