
from http import HTTPStatus
from typing import Mapping
from flask import Response, render_template, request, url_for, redirect
from flask.views import MethodView
from celery.canvas import chain
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata, 
    EntryPoint, 
    PluginMetadata, 
    PluginMetadataSchema, 
    PluginType, 
    InputDataMetadata
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import ROUTER_BLP, Router
from .schemas import InputParametersSchema
from .tasks import route_task

@ROUTER_BLP.route("/")
class PluginsView(MethodView):

    @ROUTER_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return PluginMetadata(
            title="Router",
            description=Router.instance.description,
            name=Router.instance.name,
            version=Router.instance.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{ROUTER_BLP.name}.{ProcessView.__name__}"),
                ui_href=url_for(f"{ROUTER_BLP.name}.{MicroFrontend.__name__}"),
                data_input=[
                    InputDataMetadata(
                        data_type="entity/list",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesUrl",
                    ),
                    InputDataMetadata(
                        data_type="entity/attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                        parameter="entitiesMetadataUrl",
                    ),
                    InputDataMetadata(
                        data_type="graph/taxonomy",
                        content_type=["application/zip"],
                        required=True,
                        parameter="taxonomiesZipUrl",
                    ),
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/list", 
                        content_type=["text/csv"], 
                        required=True,
                    ),
                    DataMetadata(
                        data_type="graph/taxonomy", 
                        content_type=["application/zip"], 
                        required=True,
                        ),
                    DataMetadata(
                        data_type="graph/taxonomy", 
                        content_type=["application/zip"], 
                        required=True,
                    ),
                ],
            ),
            tags=Router.instance.tags,
        )

@ROUTER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the router plugin."""

    @ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the router plugin."
    )
    @ROUTER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ), 
        location="query", 
        required=False,
    )
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors, valid=False)

    @ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the router plugin."
    )
    @ROUTER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ), 
        location="form", 
        required=False,
    )
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors, not errors)


    def render(self, data: Mapping, errors: dict, valid: bool):
        return Response(
            render_template(
                "simple_template.html",
                name=Router.instance.name,
                version=Router.instance.version,
                schema=InputParametersSchema(),
                values=data,
                valid=valid,
                errors=errors,
                process=url_for(f"{ROUTER_BLP.name}.ProcessView")
            )
        )

@ROUTER_BLP.route("/process/")
class ProcessView(MethodView):
    @ROUTER_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @ROUTER_BLP.response(HTTPStatus.SEE_OTHER)
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        db_task = ProcessingTask(
            task_name=route_task.name,
            parameters=InputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)
        
        task: chain = route_task.s(db_id=db_task.id) | save_task_result.s(db_id=db_task.id)
        
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for(
                "tasks-api.TaskView", 
                task_id=str(db_task.id)
            ), 
            HTTPStatus.SEE_OTHER
        )