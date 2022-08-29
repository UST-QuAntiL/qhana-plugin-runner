from http import HTTPStatus
from json import dumps
from typing import Mapping

from celery import chain
from flask import url_for, request, Response, render_template, redirect
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import HybridAutoencoderPlugin, HA_BLP
from .tasks import hybrid_autoencoder_pennylane_task
from .schemas import (
    HybridAutoencoderTaskResponseSchema,
    HybridAutoencoderPennylaneRequestSchema,
)
from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
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
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{HA_BLP.name}.HybridAutoencoderPennylaneAPI"),
                ui_href=url_for(f"{HA_BLP.name}.MicroFrontend"),
                data_input=[
                    DataMetadata(
                        data_type="real-valued-entities",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="real-valued-entities",
                        content_type=["application/json"],
                        required=True,
                    )
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
        return self.render(request.args, errors)

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
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = HybridAutoencoderPennylaneRequestSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=HybridAutoencoderPlugin.instance.name,
                version=HybridAutoencoderPlugin.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{HA_BLP.name}.HybridAutoencoderPennylaneAPI"),
                example_values=url_for(
                    f"{HA_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )


@HA_BLP.route("/process/pennylane/")
class HybridAutoencoderPennylaneAPI(MethodView):
    """Start a long running processing task."""

    @HA_BLP.response(HTTPStatus.OK, HybridAutoencoderTaskResponseSchema)
    @HA_BLP.arguments(
        HybridAutoencoderPennylaneRequestSchema(unknown=EXCLUDE), location="form"
    )
    @HA_BLP.require_jwt("jwt", optional=True)
    def post(self, req_dict):
        """Start the demo task."""
        db_task = ProcessingTask(
            task_name=hybrid_autoencoder_pennylane_task.name,
            parameters=dumps(req_dict),
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
