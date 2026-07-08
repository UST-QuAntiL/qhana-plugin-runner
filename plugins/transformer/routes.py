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

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
    InputDataMetadata,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import ELEMENT_TRANSFORMERS_BLP, ElementTransformers
from .schemas import InputParametersSchema
from .tasks import calculation_task


@ELEMENT_TRANSFORMERS_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @ELEMENT_TRANSFORMERS_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @ELEMENT_TRANSFORMERS_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Transformers endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Element similarities to element distances transformers",
            description=ElementTransformers.instance.description,
            name=ElementTransformers.instance.name,
            version=ElementTransformers.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{ELEMENT_TRANSFORMERS_BLP.name}.CalcSimilarityView"),
                ui_href=url_for(f"{ELEMENT_TRANSFORMERS_BLP.name}.MicroFrontend"),
                data_input=[
                    InputDataMetadata(
                        data_type="relation/element-similarities",
                        content_type=["application/zip"],
                        required=True,
                        parameter="similaritiesUrl",
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="relation/element-distances",
                        content_type=["application/zip"],
                        required=True,
                    )
                ],
            ),
            tags=ElementTransformers.instance.tags,
        )


@ELEMENT_TRANSFORMERS_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the Sym Max Mean plugin."""

    example_inputs = {
        "inputStr": "Sample input string.",
    }

    @ELEMENT_TRANSFORMERS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the similarity to distance transformers plugin.",
    )
    @ELEMENT_TRANSFORMERS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @ELEMENT_TRANSFORMERS_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, False)

    @ELEMENT_TRANSFORMERS_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend of the similarity to distance transformers plugin.",
    )
    @ELEMENT_TRANSFORMERS_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @ELEMENT_TRANSFORMERS_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        schema = InputParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=ElementTransformers.instance.name,
                version=ElementTransformers.instance.version,
                schema=schema,
                values=data,
                valid=valid,
                errors=errors,
                process=url_for(f"{ELEMENT_TRANSFORMERS_BLP.name}.CalcSimilarityView"),
                example_values=url_for(
                    f"{ELEMENT_TRANSFORMERS_BLP.name}.MicroFrontend",
                    **self.example_inputs,
                ),
            )
        )


@ELEMENT_TRANSFORMERS_BLP.route("/process/")
class CalcSimilarityView(MethodView):
    """Start a long running processing task."""

    @ELEMENT_TRANSFORMERS_BLP.arguments(
        InputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @ELEMENT_TRANSFORMERS_BLP.response(HTTPStatus.SEE_OTHER)
    @ELEMENT_TRANSFORMERS_BLP.require_jwt("jwt", optional=True)
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
