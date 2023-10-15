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
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
import os

from . import QNN_BLP, QNN
from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
    InputDataMetadata,
)
from .schemas import (
    QuantumBackends,
    OptimizerEnum,
    WeightInitEnum,
    QNNParametersSchema,
    TaskResponseSchema,
)

from .tasks import calculation_task
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result


@QNN_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QNN_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @QNN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """QNN endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Quantum Neutral Network (QNN)",
            description="Simple QNN with variable number of variational quantum layers",
            name=QNN.instance.identifier,
            version=QNN.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{QNN_BLP.name}.ProcessView"),
                ui_href=url_for(f"{QNN_BLP.name}.MicroFrontend"),
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
                        data_type="entity/label",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="trainLabelPointsUrl",
                    ),
                    InputDataMetadata(
                        data_type="entity/vector",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=True,
                        parameter="testPointsUrl",
                    ),
                    InputDataMetadata(
                        data_type="entity/label",
                        content_type=[
                            "application/json",
                            "text/csv",
                        ],
                        required=False,
                        parameter="testLabelPointsUrl",
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
                        data_type="qnn-weights",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="representative-circuit",
                        content_type=["application/qasm"],
                        required=False,
                    ),
                ],
            ),
            tags=["neural-network", "machine-learning"],
        )


@QNN_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the QNN plugin."""

    @QNN_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the QNN plugin.")
    @QNN_BLP.arguments(
        QNNParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QNN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @QNN_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the QNN plugin.")
    @QNN_BLP.arguments(
        QNNParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QNN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = QNNParametersSchema()

        data_dict = dict(data)
        # define default values
        default_values = {
            schema.fields[
                "device"
            ].data_key: QuantumBackends.aer_statevector_simulator.value,
            schema.fields["shots"].data_key: 1000,
            schema.fields[
                "optimizer"
            ].data_key: OptimizerEnum.adam.value,  # why not default in GUI?
            schema.fields["lr"].data_key: 0.07,
            schema.fields["n_qubits"].data_key: 5,
            schema.fields["resolution"].data_key: 80,
            schema.fields["epochs"].data_key: 2,
            schema.fields["q_depth"].data_key: 5,
            schema.fields["batch_size"].data_key: 10,
            schema.fields["randomly_shuffle"].data_key: True,
            schema.fields["visualize"].data_key: True,
            schema.fields["weight_init"].data_key: WeightInitEnum.uniform.value,
            schema.fields["weights_to_wiggle"].data_key: 0,
        }

        if "IBMQ_BACKEND" in os.environ:
            default_values[schema.fields["device"].data_key] = os.environ["IBMQ_BACKEND"]

        if "IBMQ_TOKEN" in os.environ:
            default_values[schema.fields["ibmq_token"].data_key] = "****"

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=QNN.instance.name,
                version=QNN.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{QNN_BLP.name}.ProcessView"),
                frontendjs=url_for(f"{QNN_BLP.name}.get_frontend_js"),
            )
        )


@QNN_BLP.route("/ui/frontend_js/")
def get_frontend_js():
    return send_file(Path(__file__).parent / "frontend.js", mimetype="text/javascript")


@QNN_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @QNN_BLP.arguments(QNNParametersSchema(unknown=EXCLUDE), location="form")
    @QNN_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @QNN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=QNNParametersSchema().dumps(arguments),
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
