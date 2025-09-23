from http import HTTPStatus

from flask.helpers import url_for
from flask.views import MethodView

from qhana_plugin_runner.api.plugin_schemas import (
    ApiLink,
    EntryPoint,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)

from .plugin import PATTERN_EDITOR_BLP, WorkflowPatternEditor


@PATTERN_EDITOR_BLP.route("/")
class PluginRootView(MethodView):
    """Plugin for editing workflow patterns."""

    @PATTERN_EDITOR_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @PATTERN_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Endpoint returning the plugin metadata"""

        return PluginMetadata(
            title="Workflow Pattern Editor",
            description="Edit workflow patterns and integrate with Pattern Atlas.",
            name=WorkflowPatternEditor.instance.name,
            version=WorkflowPatternEditor.instance.version,
            type=PluginType.interaction,
            entry_point=EntryPoint(
                href=url_for(
                    f"{PATTERN_EDITOR_BLP.name}.{PatternEditorProcess.__name__}",
                    _external=True,
                ),
                ui_href=url_for(
                    f"{PATTERN_EDITOR_BLP.name}.{PatternEditorFrontend.__name__}",
                    _external=True
                ),
                data_input=[],
                data_output=[],
            ),
            tags=WorkflowPatternEditor.instance.tags,
            links=[
                ApiLink(
                    type="patterns",
                    href=url_for(
                        f"{PATTERN_EDITOR_BLP.name}.{PatternListView.__name__}",
                        _external=True,
                    ),
                )
            ],
        )


@PATTERN_EDITOR_BLP.route("/pattern-editor-ui/")
class PatternEditorFrontend(MethodView):
    """Micro frontend for the pattern editor plugin."""

    @PATTERN_EDITOR_BLP.html_response(
        HTTPStatus.OK,
        description="Micro frontend for the pattern editor plugin.",
    )
    @PATTERN_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Return the micro frontend."""
        return """
        <html>
        <head><title>Workflow Pattern Editor</title></head>
        <body>
            <h1>Workflow Pattern Editor</h1>
            <p>This is your pattern editor interface.</p>
            <div id="pattern-editor">
                <!-- Pattern editor will be loaded here -->
            </div>
        </body>
        </html>
        """


@PATTERN_EDITOR_BLP.route("/pattern-editor/")
class PatternEditorProcess(MethodView):
    """Pattern editor process endpoint."""

    @PATTERN_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return {"status": "Pattern editor ready"}


@PATTERN_EDITOR_BLP.route("/patterns/")
class PatternListView(MethodView):
    """List all available patterns."""

    @PATTERN_EDITOR_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """Get available patterns from Pattern Atlas"""
        # TODO: Implement Pattern Atlas integration
        return {
            "patterns": [],
            "links": []
        }