from http import HTTPStatus
from typing import Mapping

from celery.canvas import chain
from celery.result import AsyncResult
from flask import Response
from flask import redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from plugins.quantum_k_means import QKMEANS_BLP, QKMeans
from plugins.quantum_k_means.schemas import InputParametersSchema, TaskResponseSchema
from qhana_plugin_runner.api.plugin_schemas import PluginMetadataSchema
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from plugins.quantum_k_means.tasks import calculation_task


@QKMEANS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QKMEANS_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @QKMEANS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Quantum k-means endpoint returning the plugin metadata."""
        return {
            "name": QKMeans.instance.name,
            "version": QKMeans.instance.version,
            "identifier": QKMeans.instance.identifier,
            "root_href": url_for(f"{QKMEANS_BLP.name}.PluginsView"),
            "title": "Quantum k-means",
            "description": "K-means algorithms that can run on quantum computers.",
            "plugin_type": "points-to-clusters",
            "tags": [],
            "processing_resource_metadata": {
                "href": url_for(f"{QKMEANS_BLP.name}.CalcView"),
                "ui_href": url_for(f"{QKMEANS_BLP.name}.MicroFrontend"),
                "inputs": [
                    [
                        {
                            "output_type": "entity-points",
                            "content_type": "application/json",
                            "name": "Entity points",
                        },
                    ]
                ],
                "outputs": [
                    [
                        {
                            "output_type": "clusters",
                            "content_type": "application/json",
                            "name": "Clusters",
                        }
                    ]
                ],
            },
        }


@QKMEANS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the quantum k-means plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @QKMEANS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the quantum k-means plugin.",
    )
    @QKMEANS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QKMEANS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @QKMEANS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the quantum k-means plugin.",
    )
    @QKMEANS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QKMEANS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)
        fields = InputParametersSchema().fields

        # define default values
        default_values = {fields["clusters_cnt"].data_key: 2}

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=QKMeans.instance.name,
                version=QKMeans.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{QKMEANS_BLP.name}.CalcView"),
                example_values=url_for(
                    f"{QKMEANS_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@QKMEANS_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @QKMEANS_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @QKMEANS_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @QKMEANS_BLP.require_jwt("jwt", optional=True)
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
        result: AsyncResult = task.apply_async()

        db_task.task_id = result.id
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(result.id)), HTTPStatus.SEE_OTHER
        )
