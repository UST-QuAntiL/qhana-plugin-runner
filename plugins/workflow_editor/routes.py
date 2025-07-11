from http import HTTPStatus
from importlib import resources
from typing import Mapping
from flask import current_app
from flask import render_template
from flask.globals import request
from flask.helpers import url_for
from flask.views import MethodView
from flask.wrappers import Response
from flask_smorest import abort
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema

from . import assets
from .plugin import WF_EDITOR_BLP, WorkflowEditor


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
