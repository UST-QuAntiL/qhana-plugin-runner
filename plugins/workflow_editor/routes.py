from datetime import datetime, timezone
from http import HTTPStatus
from importlib import resources
from typing import Literal, Mapping
from uuid import uuid4

from flask import current_app, render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    ApiLink,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema
from qhana_plugin_runner.db.models.virtual_plugins import DataBlob, PluginState

from . import assets
from .config import WF_STATE_KEY
from .plugin import WF_EDITOR_BLP, WorkflowEditor
from .schemas import WorkflowSaveParamsSchema, WorkflowSchema
from .tasks import deploy_workflow
from .util import extract_wf_properties


class WorkflowEditorSchema(FrontendFormBaseSchema):
    workflow_url = FileUrl(
        required=False,
        allow_none=True,
        data_content_types=["application/bpmn", "application/bpmn+xml"],
    )


@WF_EDITOR_BLP.route("/")
class PluginRootView(MethodView):
    """Plugin for editing BPMN workflows."""

    @WF_EDITOR_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata"""

        return PluginMetadata(
            title="Workflow Editor",
            description="Edit BPMN workflows with an online editor.",
            name=WorkflowEditor.instance.name,
            version=WorkflowEditor.instance.version,
            type=PluginType.interaction,
            entry_point=EntryPoint(
                href=url_for(
                    f"{WF_EDITOR_BLP.name}.{WorkflowEditorProcess.__name__}",
                    _external=True,
                ),
                ui_href=url_for(
                    f"{WF_EDITOR_BLP.name}.{WFEditorFrontend.__name__}", _external=True
                ),
                data_input=[],
                data_output=[],
            ),
            tags=WorkflowEditor.instance.tags,
            links=[
                ApiLink(
                    "workflows",
                    url_for(
                        f"{WF_EDITOR_BLP.name}.{WorkflowListView.__name__}",
                        _external=True,
                    ),
                )
            ],
        )


@WF_EDITOR_BLP.route("/wf-editor-ui/")
class WFEditorFrontend(MethodView):
    """Micro frontend for the workflow editor plugin."""

    @WF_EDITOR_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the workflow editor plugin.",
    )
    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""

        return self.render({}, {}, True)

    @WF_EDITOR_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the workflow editor plugin.",
    )
    @WF_EDITOR_BLP.arguments(
        WorkflowEditorSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with pre-rendered inputs."""
        return self.render(request.form, errors, not errors)

    def render(self, data: Mapping, errors: dict, valid: bool):
        plugin = WorkflowEditor.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        config = plugin.get_config()

        schema = WorkflowEditorSchema()
        return Response(
            render_template(
                "workflow-editor.html",
                name=plugin.name,
                version=plugin.version,
                schema=schema,
                valid=valid,
                values=data,
                errors=errors,
                process=url_for(f"{WF_EDITOR_BLP.name}.{WorkflowEditorProcess.__name__}"),
                wf_editor_js=url_for(
                    f"{WF_EDITOR_BLP.name}.{WorkflowEditorJavaScript.__name__}"
                ),
                wf_list_url=url_for(f"{WF_EDITOR_BLP.name}.{WorkflowListView.__name__}"),
                **config,
            )
        )


@WF_EDITOR_BLP.route("/wf-editor/")
class WorkflowEditorProcess(MethodView):
    """TODO."""

    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return ""


def _get_resource(filename):
    package = resources.files(assets)
    for r in package.iterdir():
        if r.is_dir():
            continue
        if r.is_file() and r.name == filename:
            return r
    return None


@WF_EDITOR_BLP.route("/workflow-editor.js")
class WorkflowEditorJavaScript(MethodView):
    """Get the Javascript."""

    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        script = _get_resource("editor-bundle.js")
        if script:
            return script.read_text()
        abort(HTTPStatus.NOT_FOUND)


@WF_EDITOR_BLP.route("/workflow-editor.js.map")
class WorkflowEditorJavaScriptMap(MethodView):
    """Get the Javascript sourcemap."""

    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        script = _get_resource("editor-bundle.js.map")
        if script:
            return script.read_text()
        abort(HTTPStatus.NOT_FOUND)


@WF_EDITOR_BLP.route("/workflows/")
class WorkflowListView(MethodView):
    """List all saved workflows."""

    @WF_EDITOR_BLP.response(HTTPStatus.OK, WorkflowSchema(many=True))
    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        plugin = WorkflowEditor.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        workflows = PluginState.get_value(plugin.name, WF_STATE_KEY, [])
        if workflows is None:
            workflows = []
        # TODO: find better workaround for date (de-)serialization (e.g., with custom ma field)
        workflows = [unparse_datetime(w) for w in workflows]
        return workflows

    @WF_EDITOR_BLP.arguments(WorkflowSaveParamsSchema(), location="query", as_kwargs=True)
    @WF_EDITOR_BLP.response(HTTPStatus.OK, WorkflowSchema())
    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def post(
        self, autosave: bool = False, deploy: Literal["", "plugin", "workflow"] = ""
    ):
        plugin = WorkflowEditor.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)
        if request.content_length > 100_000_000:
            abort(HTTPStatus.BAD_REQUEST, "Request body is too big!")
        bpmn = request.get_data(as_text=True)
        id_, name, version = extract_wf_properties(bpmn)
        wf_id = str(uuid4())
        workflow = {
            "id": id_,
            "version": version,
            "name": name,
            "date": datetime.now(timezone.utc).isoformat(sep="T"),
            "autosave": autosave,
            "workflow_id": wf_id,
        }
        workflows = PluginState.get_value(plugin.name, WF_STATE_KEY, [])
        if workflows is None:
            workflows = []
        DataBlob.set_value(plugin.name, wf_id, bpmn.encode(encoding="utf-8"))
        PluginState.set_value(
            plugin.name, WF_STATE_KEY, [workflow] + workflows, commit=True
        )

        if deploy:
            workflow_url = url_for(
                f"{WF_EDITOR_BLP.name}.{WorkflowView.__name__}",
                wf_id=wf_id,
                _external=True,
            )
            deploy_workflow.s(workflow_url, wf_id, deploy_as=deploy).apply_async()

        # TODO: start a cleanup task to reduce the autosaves to the last 3 saves
        # of the newest version for each workflow.

        return unparse_datetime(workflow)


def unparse_datetime(workflow: dict):
    return {
        "id": workflow["id"],
        "version": workflow["version"],
        "name": workflow["name"],
        "date": datetime.fromisoformat(workflow["date"]),
        "autosave": workflow["autosave"],
        "workflow_id": workflow["workflow_id"],
    }


@WF_EDITOR_BLP.route("/workflows/<string:wf_id>/")
class WorkflowView(MethodView):
    """Get a specific workflow file."""

    @WF_EDITOR_BLP.response(HTTPStatus.OK, content_type="application/bpmn+xml")
    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self, wf_id: str):
        plugin = WorkflowEditor.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        bpmn = DataBlob.get_value(plugin.name, wf_id, default=None)
        if bpmn is None:
            print("\n\n", wf_id, bpmn, "saad", "\n\n")
            abort(HTTPStatus.NOT_FOUND, "BAAD")

        return Response(bpmn, status=HTTPStatus.OK, content_type="application/bpmn+xml")

    @WF_EDITOR_BLP.response(HTTPStatus.OK)
    @WF_EDITOR_BLP.require_jwt("jwt", optional=True)
    def delete(self, wf_id: str):
        plugin = WorkflowEditor.instance
        if plugin is None:
            abort(HTTPStatus.INTERNAL_SERVER_ERROR)

        DataBlob.delete_value(plugin.name, wf_id)

        workflows = PluginState.get_value(plugin.name, WF_STATE_KEY)
        if workflows is None:
            workflows = []
        filtered_workflows = [w for w in workflows if w["workflow_id"] != wf_id]

        PluginState.set_value(plugin.name, WF_STATE_KEY, filtered_workflows, commit=True)
