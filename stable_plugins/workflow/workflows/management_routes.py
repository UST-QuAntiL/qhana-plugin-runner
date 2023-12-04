import json
from datetime import datetime
from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional, Sequence, Union, cast

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import redirect, render_template
from flask.globals import current_app, request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from marshmallow import INCLUDE
from requests.exceptions import HTTPError, RequestException

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.virtual_plugins import (
    VIRTUAL_PLUGIN_CREATED,
    VIRTUAL_PLUGIN_REMOVED,
    PluginState,
    VirtualPlugin,
)
from qhana_plugin_runner.tasks import save_task_error

from .clients.camunda_client import CamundaClient, CamundaManagementClient
from .datatypes.camunda_datatypes import WorkflowIncident
from .management import WORKFLOW_MGMNT_BLP, WorkflowManagement
from .schemas import AnyInputSchema, GenericInputsSchema, WorkflowIncidentSchema
from .tasks import process_input, start_workflow_with_arguments
from .watchers.workflow_status import workflow_status_watcher

config = WorkflowManagement.instance.config

TASK_LOGGER = get_task_logger(__name__)


class WorkflowIncidentWithDate(WorkflowIncident):
    incidentDatetime: datetime


def add_deployed_info(
    process_definition: Union[Dict[str, Any], Sequence[Dict[str, Any]]]
):
    if isinstance(process_definition, dict):
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            process_definition_id=process_definition.get("id"),
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
            process_definition_id=proc_def.get("id"),
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
            description="Manage workflows deployed in Camunda.",
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
            tags=["workflow", "bpmn", "camunda-engine"],
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
        try:
            camunda = CamundaManagementClient(config)
            process_definitions = camunda.get_process_definitions()
            camunda_online = True
        except RequestException:
            process_definitions = []
            camunda_online = False

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
                camunda_online=camunda_online,  # TODO use this information in the template
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


@WORKFLOW_MGMNT_BLP.route(
    "/workflows/<string:process_definition_id>/"
)  # FIXME wrong variable name
class WorkflowView(MethodView):
    """Manage a workflow instance."""

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, process_definition_id: str):
        camunda = CamundaManagementClient(config)
        try:
            process_def = camunda.get_process_definition(
                definition_id=process_definition_id
            )
        except RequestException as err:
            if isinstance(err, HTTPError) and err.response.status_code == 404:
                abort(404, message="Process definition does not exist.")
            abort(500, message="Could not reach camunda to verify process instance id.")
        add_deployed_info(process_definition=process_def)
        return process_def

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def post(self, process_definition_id: str):
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            process_definition_id=process_definition_id,
            _external=True,
        )

        if VirtualPlugin.exists([VirtualPlugin.href == plugin_url]):
            return self.get(process_definition_id)

        camunda = CamundaManagementClient(config)
        try:
            process_instance = camunda.get_process_definition(
                definition_id=process_definition_id
            )
        except RequestException as err:
            if isinstance(err, HTTPError) and err.response.status_code == 404:
                abort(404, message="Process definition does not exist.")
            abort(500, message="Could not reach camunda to verify process instance id.")

        version: str = str(process_instance.get("version", 1))
        description: Optional[str] = process_instance.get("description")
        key: Optional[str] = process_instance.get("key")

        plugin = VirtualPlugin(
            parent_id=WorkflowManagement.instance.identifier,
            name=key if key else process_definition_id,
            version=version,
            tags="\n".join(["workflow", "bpmn"]),
            description=description if description else "",
            href=plugin_url,
        )

        variables = camunda.get_workflow_start_form_variables(process_definition_id)
        if variables:
            PluginState.set_value(
                WorkflowManagement.instance.identifier, plugin_url, variables
            )

        DB.session.add(plugin)
        DB.session.commit()

        VIRTUAL_PLUGIN_CREATED.send(
            current_app._get_current_object(), plugin_url=plugin_url
        )

        return self.get(process_definition_id)

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def delete(self, process_definition_id: str):
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            process_definition_id=process_definition_id,
            _external=True,
        )

        VirtualPlugin.delete_by_href(plugin_url, WorkflowManagement.instance.identifier)

        PluginState.delete_value(WorkflowManagement.instance.identifier, plugin_url)

        DB.session.commit()

        return self.get(process_definition_id)


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

        VIRTUAL_PLUGIN_REMOVED.send(
            current_app._get_current_object(), plugin_url=plugin_url
        )

        return Response(status=HTTPStatus.NO_CONTENT)


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:process_definition_id>/bpmn/")
class WorkflowBPMNView(MethodView):
    """Manage a workflow instance."""

    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, process_definition_id: str):
        camunda = CamundaManagementClient(config)
        try:
            process_xml: str = camunda.get_process_definition_xml(
                definition_id=process_definition_id
            )
        except RequestException as err:
            if isinstance(err, HTTPError) and err.response.status_code == 404:
                abort(404, message="Process definition does not exist.")
            abort(500, message="Could not reach camunda to verify process instance id.")
        return Response(process_xml, status=HTTPStatus.OK, content_type="application/xml")


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:process_definition_id>/plugin/")
class VirtualPluginView(MethodView):
    """Metadata endpoint for a virtual workflow plugin."""

    @WORKFLOW_MGMNT_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, process_definition_id: str):
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            process_definition_id=process_definition_id,
            _external=True,
        )
        try:
            camunda = CamundaManagementClient(config)
            process_definition = camunda.get_process_definition(
                definition_id=process_definition_id
            )
        except RequestException:
            abort(
                HTTPStatus.NOT_FOUND,
                message="The BPMN process this plugin depends on was not found. Please check if Camunda is online!",
            )
        plugin = VirtualPlugin.get_by_href(
            plugin_url, WorkflowManagement.instance.identifier
        )

        title = process_definition.get("name")
        if not title:
            if plugin and plugin.name:
                title = plugin.name
            else:
                title = process_definition["id"]

        return PluginMetadata(
            title=title,
            description=plugin.description
            if plugin
            else process_definition.get("description", ""),
            name=plugin.name
            if plugin and plugin.name
            else process_definition.get("key", process_definition_id),
            version=plugin.version
            if plugin and plugin.version is not None
            else str(process_definition.get("version", 1)),
            tags=plugin.tag_list if plugin else ["workflow", "bpmn"],
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(
                    f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginProcess.__name__}",
                    process_definition_id=process_definition_id,
                    _external=True,
                ),
                ui_href=url_for(
                    f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginUi.__name__}",
                    process_definition_id=process_definition_id,
                    _external=True,
                ),
                data_input=[],
                data_output=[],
            ),
        )


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:process_definition_id>/plugin/ui/")
class VirtualPluginUi(MethodView):
    """Micro frontend for a virtual plugin."""

    @WORKFLOW_MGMNT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for a virtual workflow plugin."
    )
    @WORKFLOW_MGMNT_BLP.arguments(
        GenericInputsSchema(
            partial=True, unknown=INCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, process_definition_id: str):
        return self.render(request.args, process_definition_id, errors)

    @WORKFLOW_MGMNT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend for a virtual workflow plugin."
    )
    @WORKFLOW_MGMNT_BLP.arguments(
        GenericInputsSchema(
            partial=True, unknown=INCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, process_definition_id: str):
        return self.render(request.form, process_definition_id, errors)

    def render(self, data: Mapping, process_definition_id: str, errors: dict):
        camunda = CamundaManagementClient(config)

        try:
            process_xml: str = camunda.get_process_definition_xml(
                definition_id=process_definition_id
            )
        except RequestException:
            current_app.logger.warning(
                "Could not load workflow process definition.", exc_info=True
            )
            process_xml = ""

        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            process_definition_id=process_definition_id,
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
                    process_definition_id=process_definition_id,
                ),
            )
        )


@WORKFLOW_MGMNT_BLP.route("/workflows/<string:process_definition_id>/plugin/process/")
class VirtualPluginProcess(MethodView):
    """Micro frontend for a virtual plugin."""

    def _map_variables(self, process_definition_id: str, arguments: Mapping):
        plugin_url = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{VirtualPluginView.__name__}",
            process_definition_id=process_definition_id,
            _external=True,
        )

        form_params = cast(
            dict,
            PluginState.get_value(WorkflowManagement.instance.identifier, plugin_url, {}),
        )

        mapped_args = {
            key: {"value": val, "type": form_params[key]["type"]}
            for key, val in arguments.items()
        }
        return mapped_args

    @WORKFLOW_MGMNT_BLP.arguments(AnyInputSchema(), location="form")
    @WORKFLOW_MGMNT_BLP.response(HTTPStatus.SEE_OTHER)
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: Mapping, process_definition_id: str):
        db_task = ProcessingTask(
            task_name=start_workflow_with_arguments.name,
            parameters=json.dumps(self._map_variables(process_definition_id, arguments)),
        )
        db_task.save(commit=False)
        DB.session.flush()  # flsuh to DB to get db_task id populated

        assert isinstance(db_task.data, dict)

        db_task.data["href"] = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{HumanTaskProcessView.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["ui_href"] = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{HumanTaskFrontend.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        db_task.data["href_incident"] = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{IncidentsProcessView.__name__}",
            db_id=db_task.id,
            _external=True,
        )
        db_task.data["ui_href_incident"] = url_for(
            f"{WORKFLOW_MGMNT_BLP.name}.{IncidentsFrontend.__name__}",
            db_id=db_task.id,
            _external=True,
        )

        db_task.save(commit=True)

        task: chain = start_workflow_with_arguments.s(
            db_id=db_task.id, workflow_id=process_definition_id
        ) | workflow_status_watcher.si(db_id=db_task.id)
        task.link_error(save_task_error.s(db_id=db_task.id))

        task.apply_async()
        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


@WORKFLOW_MGMNT_BLP.route("/running/<int:db_id>/incidents-ui/")
class IncidentsFrontend(MethodView):
    """Micro frontend of a workflow incident."""

    @WORKFLOW_MGMNT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of a workflow incident."
    )
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, db_id: int):
        """Return the micro frontend."""
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        assert isinstance(db_task.data, dict)

        process_definition_id = db_task.data["camunda_process_definition_id"]
        process_instance_id = db_task.data["camunda_process_instance_id"]
        assert isinstance(process_definition_id, str)
        assert isinstance(process_instance_id, str)

        camunda = CamundaManagementClient(config)
        try:
            process_xml: str = camunda.get_process_definition_xml(
                definition_id=process_definition_id
            )
        except RequestException:
            current_app.logger.warning(
                "Could not load workflow process definition.", exc_info=True
            )
            process_xml = ""

        camunda_client = CamundaClient(config=config)

        incidents = camunda_client.get_workflow_incidents(
            process_instance_id=process_instance_id
        )
        incidents = cast(Sequence[WorkflowIncidentWithDate], incidents)

        for incident in incidents:
            incident["incidentDatetime"] = datetime.strptime(
                incident["incidentTimestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"
            )

        if not incidents:
            incidents = []

        return Response(
            render_template(
                "workflow_incident.html",
                workflow_xml=process_xml,
                incidents=incidents,
                name="Workflow Incidents Report",
                version=WorkflowManagement.instance.version,
                schema=None,
                valid=True,
                values={},
                errors={},
                process=url_for(
                    f"{WORKFLOW_MGMNT_BLP.name}.IncidentsProcessView", db_id=db_id
                ),
            )
        )


@WORKFLOW_MGMNT_BLP.route("/running/<int:db_id>/incidents/")
class IncidentsProcessView(MethodView):
    """Start a long running processing task."""

    @WORKFLOW_MGMNT_BLP.arguments(WorkflowIncidentSchema(), location="form")
    @WORKFLOW_MGMNT_BLP.response(HTTPStatus.SEE_OTHER)
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments: dict, db_id: int):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        assert isinstance(db_task.data, dict)

        process_instance_id = db_task.data["camunda_process_instance_id"]
        assert isinstance(process_instance_id, str)

        if db_task.is_finished:
            # do nothing if task is already finished!
            return redirect(
                url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
            )

        camunda_client = CamundaClient(config=config)

        if arguments.get("incident_id"):
            # TODO handle a specific incident
            return redirect(
                url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
            )
        else:
            if arguments["action"] == "continue":
                if not camunda_client.get_workflow_incidents(
                    process_instance_id=process_instance_id
                ):
                    db_task.clear_previous_step()
                    db_task.add_task_log_entry(
                        "Continuing with the workflow.", commit=True
                    )
                else:
                    # cannot continue until all incidents are resolved!
                    return redirect(
                        url_for("tasks-api.TaskView", task_id=str(db_id)),
                        HTTPStatus.SEE_OTHER,
                    )
            if arguments["action"] == "cancel":
                camunda_client.cancel_running_workflow(
                    process_instance_id=process_instance_id
                )
                db_task.clear_previous_step()
                db_task.add_task_log_entry("Cancelled workflow!", commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = workflow_status_watcher.si(db_id=db_task.id)

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )


@WORKFLOW_MGMNT_BLP.route("/running/<int:db_id>/human-task-ui/")
class HumanTaskFrontend(MethodView):
    """Micro frontend of a workflow human task."""

    @WORKFLOW_MGMNT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of a workflow human task."
    )
    @WORKFLOW_MGMNT_BLP.arguments(
        GenericInputsSchema(
            partial=True, unknown=INCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def get(self, errors, db_id: int):
        """Return the micro frontend."""
        return self.render(request.args, db_id, errors, False)

    @WORKFLOW_MGMNT_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of a workflow human task."
    )
    @WORKFLOW_MGMNT_BLP.arguments(
        GenericInputsSchema(
            partial=True, unknown=INCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def post(self, errors, db_id: int):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, db_id, errors, not errors)

    def render(self, data: Mapping, db_id: int, errors: dict, valid: bool):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        assert isinstance(db_task.data, dict)

        process_definition_id = db_task.data["camunda_process_definition_id"]
        process_instance_id = db_task.data["camunda_process_instance_id"]
        assert isinstance(process_definition_id, str)
        assert isinstance(process_instance_id, str)

        camunda = CamundaManagementClient(config)
        try:
            process_xml: str = camunda.get_process_definition_xml(
                definition_id=process_definition_id
            )
        except RequestException:
            current_app.logger.warning(
                "Could not load workflow process definition.", exc_info=True
            )
            process_xml = ""

        external_form: Optional[str] = None

        form_key = db_task.data.get("external_form_key", "")
        assert isinstance(form_key, str)

        if form_key.startswith("embedded:"):
            try:
                external_form = camunda.get_deployed_task_form(
                    db_task.data.get("human_task_id", "")
                )
            except Exception:
                external_form = "ERROR loading external form. Try again later."

        form_params: Dict[str, dict] = {}

        if not data:
            assert isinstance(db_task.data, dict)
            try:
                form_params = json.loads(db_task.data["form_params"])
            except Exception:
                # TODO raise proper error here (will throw generic one two lines later)
                form_params = {}

        default_values = {}
        for key, val in form_params.items():
            prefix_file_url = config["workflow_conf"]["form_conf"]["file_url_prefix"]
            prefix_delimiter = config["workflow_conf"]["form_conf"]["value_separator"]
            if val["value"]:
                if not (
                    val["type"] == "String"
                    and val["value"].startswith(f"{prefix_file_url}{prefix_delimiter}")
                ):
                    default_values[key] = val["value"]

        schema = AnyInputSchema(form_params)

        return Response(
            render_template(
                "workflow_human_task.html",
                workflow_xml=process_xml,
                camunda_url=config["camunda_base_url"],
                active_task_id=db_task.data.get("human_task_id", ""),
                human_task_id=db_task.data.get("human_task_definition_key", ""),
                name=WorkflowManagement.instance.name,
                version=WorkflowManagement.instance.version,
                schema=schema,
                valid=valid,
                values=default_values,
                errors=errors,
                process=url_for(
                    f"{WORKFLOW_MGMNT_BLP.name}.HumanTaskProcessView", db_id=db_id
                ),
                external_form=external_form,
            )
        )


@WORKFLOW_MGMNT_BLP.route("/running/<int:db_id>/human-task/")
class HumanTaskProcessView(MethodView):
    """Start a long running processing task."""

    @WORKFLOW_MGMNT_BLP.arguments(AnyInputSchema(), location="form")
    @WORKFLOW_MGMNT_BLP.response(HTTPStatus.SEE_OTHER)
    @WORKFLOW_MGMNT_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments, db_id: int):
        db_task: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
        if db_task is None:
            msg = f"Could not load task data with id {db_id} to read parameters!"
            TASK_LOGGER.error(msg)
            raise KeyError(msg)

        db_task.parameters = json.dumps(arguments)
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = process_input.s(db_id=db_task.id) | workflow_status_watcher.si(
            db_id=db_task.id
        )

        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_id)), HTTPStatus.SEE_OTHER
        )
