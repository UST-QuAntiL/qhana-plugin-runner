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
                # QHAna plugin configuration
                qhanaPluginRegistryURL=current_app.config.get(
                    "PLUGIN_REGISTRY_URL", "http://localhost:5006"
                ),
                # Data flow plugin configuration
                configurationsEndpoint=current_app.config.get(
                    "SERVICE_DATA_CONFIG", "http://localhost:8000/service-task"
                ),
                # OpenTOSCA plugin configuration
                opentoscaEndpoint=current_app.config.get(
                    "OPENTOSCA_ENDPOINT", "http://localhost:1337/csars"
                ),
                wineryEndpoint=current_app.config.get(
                    "WINERY_ENDPOINT", "http://localhost:8080/winery"
                ),
                # Pattern plugin configuration
                patternAtlasEndpoint=current_app.config.get(
                    "PATTERN_ATLAS_ENDPOINT", "http://localhost:8080/pattern-atlas"
                ),
                patternAtlasUIEndpoint=current_app.config.get(
                    "PATTERN_ATLAS_UI_ENDPOINT", "http://localhost:8080/pattern-atlas"
                ),
                qcAtlasEndpoint=current_app.config.get(
                    "QC_ATLAS_ENDPOINT", "http://localhost:8080/qc-atlas"
                ),
                # QuantME plugin configuration
                nisqAnalyzerEndpoint=current_app.config.get(
                    "NISQ_ANALYZER_ENDPOINT", "http://localhost:8080/nisq-analyzer"
                ),
                nisqAnalyzerUiEndpoint=current_app.config.get(
                    "NISQ_ANALYZER_UI_ENDPOINT", "http://localhost:8080/nisq-analyzer"
                ),
                qprovEndpoint=current_app.config.get(
                    "QPROV_ENDPOINT", "http://localhost:8080/qprov"
                ),
                scriptSplitterEndpoint=current_app.config.get(
                    "SCRIPT_SPLITTER_ENDPOINT", "http://localhost:8080/script-splitter"
                ),
                scriptSplitterThreshold=current_app.config.get(
                    "SCRIPT_SPLITTER_THRESHOLD", "0.5"
                ),
                qiskitRuntimeHandlerEndpoint=current_app.config.get(
                    "QISKIT_RUNTIME_HANDLER_ENDPOINT",
                    "http://localhost:8080/qiskit-runtime-handler",
                ),
                awsRuntimeHandlerEndpoint=current_app.config.get(
                    "AWS_RUNTIME_HANDLER_ENDPOINT",
                    "http://localhost:8080/aws-runtime-handler",
                ),
                transformationFrameworkEndpoint=current_app.config.get(
                    "TRANSFORMATION_FRAMEWORK_ENDPOINT",
                    "http://localhost:8080/transformation-framework",
                ),
                # Editor Configuration
                camundaEndpoint=current_app.config.get(
                    "CAMUNDA_ENDPOINT", "http://localhost:8080/camunda"
                ),
                downloadFileName=current_app.config.get(
                    "DOWNLOAD_FILE_NAME", "workflow.bpmn"
                ),
                transformedWorkflowHandler=current_app.config.get(
                    "TRANSFORMED_WORKFLOW_HANDLER", "inline"
                ),
                autoSaveFileOption=current_app.config.get(
                    "AUTO_SAVE_FILE_OPTION", "interval"
                ),
                fileFormat=current_app.config.get("FILE_FORMAT", "bpmn"),
                autoSaveIntervalSize=current_app.config.get(
                    "AUTO_SAVE_INTERVAL", "300000"
                ),
                githubToken=current_app.config.get("GITHUB_TOKEN", ""),
                githubRepositoryName=current_app.config.get("QRM_REPONAME", ""),
                githubUsername=current_app.config.get("QRM_USERNAME", ""),
                githubRepositoryPath=current_app.config.get("QRM_REPOPATH", ""),
                uploadGithubRepositoryName=current_app.config.get(
                    "UPLOAD_GITHUB_REPO", ""
                ),
                uploadGithubRepositoryOwner=current_app.config.get(
                    "UPLOAD_GITHUB_USER", ""
                ),
                uploadGithubRepositoryPath=current_app.config.get(
                    "UPLOAD_GITHUB_REPOPATH", "qrms"
                ),
                uploadFileName=current_app.config.get(
                    "UPLOAD_FILE_NAME", "quantum-workflow-model"
                ),
                uploadBranchName=current_app.config.get("UPLOAD_BRANCH_NAME", ""),
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
