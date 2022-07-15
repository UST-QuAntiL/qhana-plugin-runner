from http import HTTPStatus
from typing import Mapping

import flask
from celery.canvas import chain
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from . import COSTUME_LOADER_BLP, CostumeLoader
from .schemas import InputParameters, InputParametersSchema, TaskResponseSchema
from .tasks import loading_task
from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result


@COSTUME_LOADER_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @COSTUME_LOADER_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Plugin loader endpoint returning the plugin metadata."""
        return PluginMetadata(
            title="Costume loader",
            description=CostumeLoader.instance.description,
            name=CostumeLoader.instance.name,
            version=CostumeLoader.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{COSTUME_LOADER_BLP.name}.LoadingView"),
                ui_href=url_for(f"{COSTUME_LOADER_BLP.name}.MicroFrontend"),
                data_input=[],
                data_output=[
                    DataMetadata(
                        data_type="raw",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="graphs",
                        content_type=["application/zip"],
                        required=True,
                    ),
                ],
            ),
            tags=CostumeLoader.instance.tags,
        )


@COSTUME_LOADER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the costume loader plugin."""

    @COSTUME_LOADER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the costume loader plugin."
    )
    @COSTUME_LOADER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @COSTUME_LOADER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the costume loader plugin."
    )
    @COSTUME_LOADER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        data_dict = dict(data)
        app = flask.current_app
        fields = InputParametersSchema().fields

        # define default values
        default_values = {
            fields["db_host"].data_key: app.config.get("COSTUME_LOADER_DB_HOST"),
            fields["db_user"].data_key: app.config.get("COSTUME_LOADER_DB_USER"),
            fields["db_password"].data_key: app.config.get("COSTUME_LOADER_DB_PASSWORD"),
            fields["db_database"].data_key: app.config.get("COSTUME_LOADER_DB_DATABASE"),
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=CostumeLoader.instance.name,
                version=CostumeLoader.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{COSTUME_LOADER_BLP.name}.LoadingView"),
            )
        )


@COSTUME_LOADER_BLP.route("/load_costumes_and_taxonomies/")
class LoadingView(MethodView):
    """Start a long running processing task."""

    @COSTUME_LOADER_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @COSTUME_LOADER_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @COSTUME_LOADER_BLP.require_jwt("jwt", optional=True)
    def post(self, input_params: InputParameters):
        """Start the costume loading task."""
        db_task = ProcessingTask(
            task_name=loading_task.name,
            parameters=InputParametersSchema().dumps(input_params),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = loading_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )
