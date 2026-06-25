
from celery.canvas import chain
from http import HTTPStatus
from flask import Response, render_template, request, url_for, redirect
from flask.views import MethodView
from marshmallow import EXCLUDE
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import ROUTER_BLP, Router
from .schemas import InputParametersSchema, MetricEnum
from .tasks import route_task, handle_webhook_task

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata, 
    EntryPoint, 
    PluginMetadata, 
    PluginMetadataSchema, 
    PluginType, 
    InputDataMetadata
)

# --- HELPER FUNCTION FOR UIs ---
def render_step(schema, data, errors, process_url):
    return Response(render_template(
        "simple_template.html", name=Router.instance.name, version=Router.instance.version,
        schema=schema, values=data, errors=errors, process=process_url
    ))

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
                    # InputDataMetadata(
                    #     data_type="entity/list",
                    #     content_type=["application/json"],
                    #     required=True,
                    #     parameter="entitiesUrl",
                    # ),
                    # InputDataMetadata(
                    #     data_type="entity/attribute-metadata",
                    #     content_type=["application/json"],
                    #     required=True,
                    #     parameter="entitiesMetadataUrl",
                    # ),
                    # InputDataMetadata(
                    #     data_type="graph/taxonomy",
                    #     content_type=["application/zip"],
                    #     required=True,
                    #     parameter="taxonomiesZipUrl",
                    # ),
                ],
                data_output=[
                    # WU-Palmer output
                    DataMetadata(data_type="custom/element-similarities", content_type=["application/zip"], required=True),
                    # Sym-Max-Mean output
                    DataMetadata(data_type="custom/attribute-similarities", content_type=["application/zip"], required=True),
                    # Transformer output
                    DataMetadata(data_type="custom/attribute-distances", content_type=["application/zip"], required=True),
                    # Aggregator output
                    DataMetadata(data_type="custom/entity-distances", content_type=["application/json"], required=True),
                    # MDS output = final output
                    DataMetadata(data_type="entity/vector", content_type=["application/json"], required=True)
                ],
            ),
            tags=Router.instance.tags,
        )

@ROUTER_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the router plugin."""

    @ROUTER_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the router plugin.")
    @ROUTER_BLP.arguments(InputParametersSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True), location="query")
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors): 
        return self.render(data=request.args, errors=errors, valid=not errors)

   
    @ROUTER_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the router plugin.")
    @ROUTER_BLP.arguments(InputParametersSchema(partial=True, unknown=EXCLUDE, validate_errors_as_result=True), location="form")
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors): 
        """Return the micro frontend with prerendered inputs."""
        return self.render(data=request.form, errors=errors, valid=not errors)
    

    def render(self, data, errors: dict, valid: bool):
        data_dict = dict(data)
        fields = InputParametersSchema().fields

        default_values = {
            fields["dimensions"].data_key: 2,
            fields["metric"].data_key: MetricEnum.metric_mds,
            fields["n_init"].data_key: 4,
            fields["max_iter"].data_key: 300,
        }
         
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=Router.instance.name,
                version=Router.instance.version,
                schema=InputParametersSchema(),
                values=data_dict,
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
        db_task = ProcessingTask(task_name=route_task.name, parameters=InputParametersSchema().dumps(arguments))
        db_task.save(commit=True)
        
        db_task.data["webhook_url"] = url_for(f"{ROUTER_BLP.name}.WebhookView", db_id=db_task.id, _external=True)
        db_task.save(commit=True)
        
        task = route_task.s(db_id=db_task.id)
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER)


# --- WEBHOOOK ---
@ROUTER_BLP.route("/<int:db_id>/webhook/")
class WebhookView(MethodView):
    """Endpoint to receive webhook updates from called plugins."""
    
    @ROUTER_BLP.response(HTTPStatus.OK)
    def post(self, db_id: int):
        source_url = request.args.get("source")
        event = request.args.get("event")
        
        if source_url and event == "status":
            handle_webhook_task.delay(db_id=db_id, source_url=source_url)
            
        return "Webhook received", HTTPStatus.OK