from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import redirect, render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from marshmallow import EXCLUDE
from requests.exceptions import HTTPError, RequestException

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.virtual_plugins import PluginState, VirtualPlugin

from .clients.camunda_client import CamundaManagementClient
from .management import WORKFLOW_MGMNT_BLP, WorkflowManagement
from .schemas import AnyInputSchema, WorkflowsParametersSchema

config = WorkflowManagement.instance.config

TASK_LOGGER = get_task_logger(__name__)


def add_deployed_info(
    process_definition: Union[Dict[str, Any], Sequence[Dict[str, Any]]]
):
    if isinstance(process_definition, dict):
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            instance_id=process_definition.get("id"),
            _external=True,
        )
        process_definition["pluginHref"] = plugin_url
        process_definition["deployed"] = VirtualPlugin.exists(
            [VirtualPlugin.href == plugin_url]
        )
        return
    plugins = VirtualPlugin.get_all(for_parents=[WorkflowManagement.instance.identifier])
    urls = {p.href for p in plugins}
    for proc_def in process_definition:
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            instance_id=proc_def.get("id"),
            _external=True,
        )
        proc_def["pluginHref"] = plugin_url
        proc_def["deployed"] = plugin_url in urls


@WORKFLOW_MGMNT_BLP.route("/")
class WfManagementView(MethodView):
    """Plugin for managing workflows deployed in camunda."""

    @WORKFLOW_MGMNT_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata"""

        return PluginMetadata(
            title="Workflow Management",
            description="Manage workflows deploayed in Camunda.",
            name=WorkflowManagement.instance.name,
            version=WorkflowManagement.instance.version,
            type=PluginType.interaction,
            entry_point=EntryPoint(
                href=url_for(f"{WORKFLOW_MGMNT_BLP.name}.WorkflowsView", _external=True),
                ui_href=url_for(
                    f"{WORKFLOW_MGMNT_BLP.name}.MicroFrontend", _external=True
                ),
                data_input=[],
                data_output=[],
            ),
            tags=["bpmn", "camunda engine"],
        )


@WORKFLOW_MGMNT_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the workflow management plugin."""

    @WORKFLOW_MGMNT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the workflow management plugin."
    )
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""
        camunda = CamundaManagementClient(config)
        process_definitions = camunda.get_process_definitions()

        add_deployed_info(process_definitions)

        potential_plugins = {w["pluginHref"] for w in process_definitions}

        plugins_wo_workflow = VirtualPlugin.get_all(
            [WorkflowManagement.instance.identifier],
            filters=[~VirtualPlugin.href.in_(potential_plugins)],
        )

        return Response(
            render_template(
                "camunda_workflows.html",
                workflows=process_definitions,
                deployEndpoint=f"{WORKFLOW_MGMNT_BLP.name}.{WorkflowView.__name__}",
                pluginUiEndpoint=f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginUi.__name__}",
                plugins_wo_workflow=plugins_wo_workflow,
                undeployEndpoint=f"{WORKFLOW_MGMNT_BLP.name}.{UndeployPluginView.__name__}",
            )
        )


@WORKFLOW_MGMNT_BLP.route("/workflows/")
class WorkflowsView(MethodView):
    """Get all workflow instances."""

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self):
        camunda = CamundaManagementClient(config)
        process_definitions = camunda.get_process_definitions()

        add_deployed_info(process_definitions)

        return process_definitions


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:instance_id>/")  # FIXME wrong variable name
class WorkflowView(MethodView):
    """Manage a workflow instance."""

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, instance_id: str):
        camunda = CamundaManagementClient(config)
        try:
            process_def = camunda.get_process_definition(definition_id=instance_id)
        except RequestException as err:
            if isinstance(err, HTTPError) and err.response.status_code == 404:
                abort(404, message="Process definition does not exist.")
            abort(500, message="Could not reach camunda to verify process instance id.")
        add_deployed_info(process_definition=process_def)
        return process_def

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def post(self, instance_id: str):
        camunda = CamundaManagementClient(config)
        try:
            process_instance = camunda.get_process_definition(definition_id=instance_id)
        except RequestException as err:
            if isinstance(err, HTTPError) and err.response.status_code == 404:
                abort(404, message="Process definition does not exist.")
            abort(500, message="Could not reach camunda to verify process instance id.")

        version: str = str(process_instance.get("version", 1))
        description: Optional[str] = process_instance.get("description")
        key: Optional[str] = process_instance.get("key")

        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            instance_id=instance_id,
            _external=True,
        )

        if VirtualPlugin.exists([VirtualPlugin.href == plugin_url]):
            return self.get(instance_id)

        plugin = VirtualPlugin(
            parent_id=WorkflowManagement.instance.identifier,
            name=key if key else instance_id,
            version=version,
            tags="\n".join(["workflow", "bpmn"]),
            description=description if description else "",
            href=url_for(
                f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
                instance_id=instance_id,
                _external=True,
            ),
        )

        variables = camunda.get_workflow_start_form_variables(instance_id)
        if variables:
            PluginState.set_value(
                WorkflowManagement.instance.identifier, plugin_url, variables
            )

        DB.session.add(plugin)
        DB.session.commit()
        return self.get(instance_id)

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def delete(self, instance_id: str):
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            instance_id=instance_id,
            _external=True,
        )

        VirtualPlugin.delete_by_href(plugin_url, WorkflowManagement.instance.identifier)

        PluginState.delete_value(WorkflowManagement.instance.identifier, plugin_url)

        DB.session.commit()

        return self.get(instance_id)


@WORKFLOW_MGMNT_BLP.route("/plugins/<string:plugin_url>/")
class UndeployPluginView(MethodView):
    """Undeploy an existing virtual plugin."""

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def delete(self, plugin_url: str):
        VirtualPlugin.delete_by_href(
            # TODO find better workaround to slash escaping/unescaping in url path!
            plugin_url.replace("%2F", "/"),
            WorkflowManagement.instance.identifier,
        )

        PluginState.delete_value(WorkflowManagement.instance.identifier, plugin_url)

        DB.session.commit()

        return Response(status=HTTPStatus.NO_CONTENT)


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:instance_id>/bpmn/")
class WorkflowBPMNView(MethodView):
    """Manage a workflow instance."""

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, instance_id: str):
        camunda = CamundaManagementClient(config)
        try:
            process_xml: str = camunda.get_process_definition_xml(
                definition_id=instance_id
            )
        except RequestException as err:
            if isinstance(err, HTTPError) and err.response.status_code == 404:
                abort(404, message="Process definition does not exist.")
            abort(500, message="Could not reach camunda to verify process instance id.")
        return Response(process_xml, status=HTTPStatus.OK, content_type="application/xml")


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:instance_id>/plugin/")
class VirtualPluginView(MethodView):
    """Metadata endpoint for a virtual workflow plugin."""

    @WORKFLOW_MGMNT_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, instance_id: str):
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            instance_id=instance_id,
            _external=True,
        )
        camunda = CamundaManagementClient(config)
        process_definition = camunda.get_process_definition(definition_id=instance_id)
        plugin = VirtualPlugin.get_by_href(
            plugin_url, WorkflowManagement.instance.identifier
        )

        return PluginMetadata(
            title=process_definition.get(
                "name", plugin.name if plugin else process_definition["id"]
            ),
            description=plugin.description
            if plugin
            else process_definition.get("description", ""),
            name=plugin.name if plugin else process_definition.get("key", instance_id),
            version=plugin.version
            if plugin
            else str(process_definition.get("version", 1)),
            tags=plugin.tag_list if plugin else ["workflow", "bpmn"],
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(
                    f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginUi.__name__}",
                    instance_id=instance_id,
                    _external=True,
                ),
                ui_href=url_for(
                    f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginProcess.__name__}",
                    instance_id=instance_id,
                    _external=True,
                ),
                data_input=[],
                data_output=[],
            ),
        )


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:instance_id>/plugin/ui/")
class VirtualPluginUi(MethodView):
    """Micro frontend for a virtual plugin."""

    @WORKFLOW_MGMNT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for a virtual workflow plugin."
    )
    @WORKFLOW_MGMNT_BLP.arguments(
        WorkflowsParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, instance_id: str):
        return self.render(request.args, instance_id, errors)

    @WORKFLOW_MGMNT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for a virtual workflow plugin."
    )
    @WORKFLOW_MGMNT_BLP.arguments(
        WorkflowsParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, instance_id: str):
        return self.render(request.form, instance_id, errors)

    def render(self, data: Mapping, instance_id: str, errors: dict):
        camunda = CamundaManagementClient(config)
        process_xml: str = camunda.get_process_definition_xml(definition_id=instance_id)

        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            instance_id=instance_id,
            _external=True,
        )
        plugin = VirtualPlugin.get_by_href(
            plugin_url, WorkflowManagement.instance.identifier
        )

        data = dict(data)

        form_params = cast(
            dict,
            PluginState.get_value(WorkflowManagement.instance.identifier, plugin_url, {}),
        )

        for key, val in form_params.items():
            prefix_file_url = config["workflow_conf"]["form_conf"]["file_url_prefix"]
            prefix_delimiter = config["workflow_conf"]["form_conf"]["value_separator"]
            if val["value"]:
                if not (
                    val["type"] == "String"
                    and val["value"].startswith(f"{prefix_file_url}{prefix_delimiter}")
                ):
                    data[key] = val["value"]

        schema = AnyInputSchema(form_params)

        return Response(
            render_template(
                "workflow_start.html",
                workflow_xml=process_xml,
                name=plugin.name if plugin else "UNKNOWN",
                version=plugin.version if plugin else "-1",
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(
                    f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginProcess.__name__}",
                    instance_id=instance_id,
                ),
            )
        )


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:instance_id>/plugin/process/")
class VirtualPluginProcess(MethodView):
    """Micro frontend for a virtual plugin."""

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(
        self, instance_id: str
    ):  # FIXME this must be a post endpoint that starts the actual workflow (see old plugin for this)
        camunda = CamundaManagementClient(config)
        return camunda.get_process_definition(definition_id=instance_id)
