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

from plugins.pca import PCA_BLP, PCA
from plugins.pca.schemas import (
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
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from plugins.pca.tasks import calculation_task


@PCA_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @PCA_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @PCA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """PCA endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Principle Component Analysis (PCA)",
            description="Reduces number of dimensions. (New ONB are the k first principle components)",
            name=PCA.instance.identifier,
            version=PCA.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{PCA_BLP.name}.ProcessView"),
                ui_href=url_for(f"{PCA_BLP.name}.MicroFrontend"),
                data_input=[
                    DataMetadata(
                        data_type="entity-points",
                        content_type=["text/csv", "application/json"],
                        required=True,
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="principle-components",
                        content_type=["application/json"],
                        required=True,
                    )
                ],
            ),
            tags=["dimension-reduction"],
        )


@PCA_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the PCA plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

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
            fields["kernel_coef"].data_key: 1
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=PCA.instance.name,
                version=PCA.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{PCA_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{PCA_BLP.name}.MicroFrontend", **self.example_inputs
                ),
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
