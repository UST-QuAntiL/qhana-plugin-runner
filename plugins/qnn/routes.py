from http import HTTPStatus
from typing import Mapping

from celery.canvas import chain
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE
import os

from plugins.qnn import QNN_BLP, QNN
from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
)
from plugins.qnn.schemas import (
    QuantumBackends,
    OptimizerEnum,
    WeightInitEnum,
    QNNParametersSchema,
    TaskResponseSchema,
)

from plugins.qnn.tasks import calculation_task
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
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
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{QNN_BLP.name}.ProcessView"),
                ui_href=url_for(f"{QNN_BLP.name}.MicroFrontend"),
                data_input=[
                    DataMetadata(
                        data_type="entity-points",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="clusters",
                        content_type=["application/json"],
                        required=True,
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="plot",
                        content_type=["text/html"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="qnn-weights",
                        content_type=["application/json"],
                        required=True,
                    ),
                ],
            ),
            tags=["neural-network", "machine-learning"],
        )


@QNN_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the QNN plugin."""

    example_inputs = {  # TODO?
        "inputStr": "Sample input string.",
    }

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
            schema.fields["test_percentage"].data_key: 0.05,
            schema.fields["device"].data_key: QuantumBackends.aer_statevector_simulator,
            schema.fields["shots"].data_key: 10,
            schema.fields["optimizer"].data_key: OptimizerEnum.adam,
            schema.fields["step"].data_key: 0.07,
            schema.fields["n_qubits"].data_key: 5,
            schema.fields["N_total_iterations"].data_key: 2,
            schema.fields["q_depth"].data_key: 5,
            schema.fields["batch_size"].data_key: 10,
            schema.fields["use_default_dataset"].data_key: False,
            schema.fields["randomly_shuffle"].data_key: True,
            schema.fields["weight_init"].data_key: WeightInitEnum.uniform,
        }

        if "IBMQ_BACKEND" in os.environ:
            default_values[schema.fields["device"].data_key] = os.environ["IBMQ_BACKEND"]

        if "IBMQ_TOKEN" in os.environ:
            default_values[schema.fields["ibmq_token"].data_key] = "****"

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        # schema.fields["use_default_dataset"].data_key: True,
        # schema.entity_points_url.data_key: "http://host.docker.internal:9090/experiments/1/data/entity_points.json/download?version=1",
        # schema.clusters_url.data_key: "http://host.docker.internal:9090/experiments/1/data/clusters.json/download?version=1",

        return Response(
            render_template(
                "simple_template.html",
                name=QNN.instance.name,
                version=QNN.instance.version,
                schema=schema,
                values=data_dict,
                errors=errors,
                process=url_for(f"{QNN_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{QNN_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


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
