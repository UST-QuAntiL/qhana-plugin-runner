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

from . import QISKIT_QKE_BLP, QiskitQKE
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


@QISKIT_QKE_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QISKIT_QKE_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @QISKIT_QKE_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Qiskit quantum kernel estimation endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Qiskit Quantum Kernel Estimation",
            description=QiskitQKE.instance.description,
            name=QiskitQKE.instance.name,
            version=QiskitQKE.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{QISKIT_QKE_BLP.name}.CalcView"),
                ui_href=url_for(f"{QISKIT_QKE_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="entityPointsUrl1",
                    ),
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="entityPointsUrl2",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="kernel-matrix",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
            ),
            tags=QiskitQKE.instance.tags,
        )


@QISKIT_QKE_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the qiskit quantum kernel estimation plugin."""

    @QISKIT_QKE_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the qiskit quantum kernel estimation plugin.",
    )
    @QISKIT_QKE_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QISKIT_QKE_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @QISKIT_QKE_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the qiskit quantum kernel estimation plugin.",
    )
    @QISKIT_QKE_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QISKIT_QKE_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)
        fields = InputParametersSchema().fields

        # define default values
        default_values = {
            fields["n_qbits"].data_key: 2,
            fields["reps"].data_key: 2,
            fields["shots"].data_key: 1024,
            fields["backend"].data_key: QiskitBackends.aer_statevector_simulator.value,
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
                name=QiskitQKE.instance.name,
                version=QiskitQKE.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{QISKIT_QKE_BLP.name}.CalcView"),
            )
        )


@QISKIT_QKE_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @QISKIT_QKE_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @QISKIT_QKE_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @QISKIT_QKE_BLP.require_jwt("jwt", optional=True)
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
