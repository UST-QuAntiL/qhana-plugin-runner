# Copyright 2026 QHAna plugin runner contributors.
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
from json import loads
from typing import Mapping, Optional

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, current_app, redirect, render_template, request, url_for
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import (
    TASK_STEPS_CHANGED,
    add_step,
    save_task_error,
)

from . import ROUTER_BLP, Router
from .schemas import (
    PIPELINE_OPTIONS,
    InputParametersSchema,
    MetricEnum,
    RoutingStepParametersSchema,
)
from .tasks import (
    PIPELINE_PLUGINS,
    handle_webhook_task,
    preprocessing_task,
    route_task,
)

TASK_LOGGER = get_task_logger(__name__)


# --- HELPER FUNCTION FOR UIs ---
def render_step(schema, data, errors, process_url):
    return Response(
        render_template(
            "simple_template.html",
            name=Router.instance.name,
            version=Router.instance.version,
            schema=schema,
            values=data,
            errors=errors,
            process=process_url,
        )
    )


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
                    # WU-Palmer output (optional)
                    DataMetadata(
                        data_type="relation/element-similarities",
                        content_type=["application/zip"],
                        required=True,
                    ),
                    # Transformer output (optional)
                    DataMetadata(
                        data_type="relation/element-distances",
                        content_type=["application/zip"],
                        required=True,
                    ),
                    # Aggregator output (optional)
                    DataMetadata(
                        data_type="relation/attribute-distances",
                        content_type=["application/zip"],
                        required=True,
                    ),
                    # MDS output = final output
                    DataMetadata(
                        data_type="entity/vector",
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
    )
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        return self.render(data=request.args, errors=errors, valid=not errors)

    @ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the router plugin."
    )
    @ROUTER_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
    )
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
                process=url_for(f"{ROUTER_BLP.name}.ProcessView"),
            )
        )


@ROUTER_BLP.route("/process/")
class ProcessView(MethodView):
    @ROUTER_BLP.arguments(InputParametersSchema(unknown=EXCLUDE), location="form")
    @ROUTER_BLP.response(HTTPStatus.SEE_OTHER)
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Discover the taxonomy attributes and queue the routing step."""
        db_task = ProcessingTask(
            task_name=route_task.name, parameters=InputParametersSchema().dumps(arguments)
        )
        db_task.save(commit=True)

        step_id = "routing-step"
        href = url_for(
            f"{ROUTER_BLP.name}.RoutingStepView", db_id=db_task.id, _external=True
        )
        ui_href = url_for(
            f"{ROUTER_BLP.name}.RoutingStepFrontend", db_id=db_task.id, _external=True
        )

        task: chain = preprocessing_task.s(db_id=db_task.id) | add_step.s(
            db_id=db_task.id, step_id=step_id, href=href, ui_href=ui_href, prog_value=50
        )
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@ROUTER_BLP.route("/<int:db_id>/routing-step-ui/")
class RoutingStepFrontend(MethodView):
    """Micro frontend for the routing step (one pipeline dropdown per attribute)."""

    @ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the router routing step."
    )
    @ROUTER_BLP.arguments(
        RoutingStepParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        return self.render(request.args, db_id, errors, False)

    @ROUTER_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the router routing step."
    )
    @ROUTER_BLP.arguments(
        RoutingStepParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        return self.render(request.form, db_id, errors, not errors)

    def render(self, data: Mapping, db_id: int, errors: dict, valid: bool):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        attributes = db_task.data.get("taxonomy_attributes", [])
        input_params = loads(db_task.parameters or "{}")

        return Response(
            render_template(
                "routing_step.html",
                name=Router.instance.name,
                version=Router.instance.version,
                schema=RoutingStepParametersSchema(),
                attributes=attributes,
                pipeline_options=PIPELINE_OPTIONS,
                input_params=input_params,
                values=data,
                valid=valid,
                errors=errors,
                process=url_for(f"{ROUTER_BLP.name}.RoutingStepView", db_id=db_id),
            )
        )


@ROUTER_BLP.route("/<int:db_id>/routing-step-process/")
class RoutingStepView(MethodView):
    """Record the per-attribute routing and launch the Wu-Palmer pipeline."""

    @ROUTER_BLP.arguments(RoutingStepParametersSchema(unknown=EXCLUDE), location="form")
    @ROUTER_BLP.response(HTTPStatus.SEE_OTHER)
    @ROUTER_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        # The routing selections are kept in ``data`` rather than merged into
        # ``parameters``. ``route_task`` reloads ``parameters`` through
        # ``InputParametersSchema`` which would reject the dynamic
        # ``pipeline_<attribute>`` fields.
        prefix = "pipeline_"
        selections = {
            key[len(prefix) :]: value
            for key, value in arguments.items()
            if key.startswith(prefix)
        }
        wu_palmer_attributes = [
            attr for attr, option in selections.items() if option == "Wu-Palmer"
        ]
        mapping_attributes = [
            attr for attr, option in selections.items() if option == "Mapping"
        ]
        one_hot_attributes = [
            attr for attr, option in selections.items() if option == "One-Hot"
        ]

        pipeline_queue = []
        if wu_palmer_attributes:
            db_task.add_task_log_entry(
                f"Queued Wu-Palmer pipeline for attributes: {wu_palmer_attributes}"
            )
            db_task.data["wu_palmer_attributes"] = "\n".join(wu_palmer_attributes)
            pipeline_queue.append("wu_palmer")

        if mapping_attributes:
            db_task.add_task_log_entry(
                f"Queued distances mapping pipeline for attributes: {mapping_attributes}"
            )
            db_task.data["mapping_attributes"] = "\n".join(mapping_attributes)
            pipeline_queue.append("mapping")
        
        if one_hot_attributes:
            db_task.add_task_log_entry(
                f"One-Hot encoding not yet supported. Selected One-Hot for attributes: {mapping_attributes}"
            )
        
        none_selected = [attr for attr, option in selections.items() if option == "None"]
        if none_selected:
            db_task.add_task_log_entry(
                f"None selected attributes skipped: {none_selected}"
            )
        
        db_task.data["pipeline_queue"] = pipeline_queue
        db_task.data["current_pipeline"] = None
      
        db_task.data["webhook_url"] = url_for(
            f"{ROUTER_BLP.name}.WebhookView", db_id=db_task.id, _external=True
        )

        # Resolve the pipeline plugin metadata urls here, where the request
        # context supplies the host. The worker has no request context and
        # SERVER_NAME is unset, so url_for(_external=True) cannot run there.
        db_task.data["plugin_urls"] = {
            key: url_for("plugins-api.PluginView", plugin=name, _external=True)
            for key, name in PIPELINE_PLUGINS.items()
        }
        db_task.clear_previous_step()
        db_task.save(commit=True)

        app = current_app._get_current_object()
        TASK_STEPS_CHANGED.send(app, task_id=db_id)

        task = route_task.s(db_id=db_task.id)
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


# --- WEBHOOOK ---
@ROUTER_BLP.route("/<int:db_id>/webhook/")
class WebhookView(MethodView):
    """Endpoint to receive webhook updates from called plugins."""

    @ROUTER_BLP.response(HTTPStatus.OK)
    def post(self, db_id: int):
        source_url = request.args.get("source")
        event = request.args.get("event")

        if source_url and event == "status":
            # Countdown prevent overload for backend.
            handle_webhook_task.apply_async(
                kwargs={"db_id": db_id, "source_url": source_url}, countdown=2
            )

        return "Webhook received", HTTPStatus.OK
