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

from . import QKNN_BLP, QKNN
from .backend.quantum_backends import QuantumBackends
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

from .frontend_js import frontend_js


@QKNN_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QKNN_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @QKNN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Quantum k nearest neighbours endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Quantum k Nearest Neighbours",
            description=QKNN.instance.description,
            name=QKNN.instance.name,
            version=QKNN.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{QKNN_BLP.name}.CalcView"),
                ui_href=url_for(f"{QKNN_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity-points",
                        content_type=["application/json"],
                        required=True,
                        parameter="entityPointsUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="labels",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
            ),
            tags=QKNN.instance.tags,
        )


@QKNN_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the quantum k nearest neighbours plugin."""

    @QKNN_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the quantum k nearest neighbours plugin.",
    )
    @QKNN_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QKNN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @QKNN_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the quantum k nearest neighbours plugin.",
    )
    @QKNN_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QKNN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)
        fields = InputParametersSchema().fields

        # define default values
        default_values = {
            fields["k"].data_key: 1,
            fields["minimize_qubit_count"].data_key: False,
            fields["exp_itr"].data_key: 10,
            fields["backend"].data_key: QuantumBackends.aer_statevector_simulator.value,
            fields["shots"].data_key: 1024,
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
                name=QKNN.instance.name,
                version=QKNN.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{QKNN_BLP.name}.CalcView"),
                frontendjs=url_for(f"{QKNN_BLP.name}.get_frontend_js"),
            )
        )


@QKNN_BLP.route("/ui/frontend_js/")
def get_frontend_js():
    return Response(frontend_js, mimetype='text/javascript')


@QKNN_BLP.route("/process/")
class CalcView(MethodView):
    """Start a long running processing task."""

    @QKNN_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @QKNN_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @QKNN_BLP.require_jwt("jwt", optional=True)
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
