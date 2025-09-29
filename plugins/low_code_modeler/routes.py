from http import HTTPStatus
from flask import render_template, send_from_directory
from flask.views import MethodView
from flask.helpers import url_for
from flask.wrappers import Response

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginMetadataSchema,
    EntryPoint,
    PluginType,
)

from .plugin import LCM_BLP, LowCodeModeler


@LCM_BLP.route("/")
class PluginRootView(MethodView):
    @LCM_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @LCM_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return PluginMetadata(
            title="LCM",
            description=LowCodeModeler.instance.description,
            name=LowCodeModeler.instance.name,
            version=LowCodeModeler.instance.version,
            tags=LowCodeModeler.instance.tags,
            type=PluginType.interaction,
            entry_point=EntryPoint(
                href=url_for(f"{LCM_BLP.name}.{Process.__name__}"),
                ui_href=url_for(f"{LCM_BLP.name}.{UI.__name__}", path="index.html"),
                plugin_dependencies=[],
                data_input=[],
                data_output=[],
            ),
        )


@LCM_BLP.route("/ui/<path:path>")
class UI(MethodView):
    @LCM_BLP.response(HTTPStatus.OK)
    @LCM_BLP.require_jwt("jwt", optional=True)
    def get(self, path):
        return send_from_directory(f"{LCM_BLP.root_path}/static", path)

    @LCM_BLP.response(HTTPStatus.OK)
    @LCM_BLP.require_jwt("jwt", optional=True)
    def post(self):
        return Response(
            """
        <script>
            /**
            * Send a message to the parent window.
            *
            * @param {string|object} message the data attribute of the created message event
            */
            function sendMessage(message) {
                var targetWindow = window.opener || window.parent;
                if (targetWindow) {
                    targetWindow.postMessage(message, "*");
                } else {
                    console.warn("Failed to message parent window. Is this page loaded outside of an iframe?");
                }
            }
                sendMessage("ui-loaded");
        </script>
        <style>
            :root {
                color: red;
                font-size: 32px;
            }
        </style>
        POST
        """
        )


@LCM_BLP.route("/process/")
class Process(MethodView):
    """TODO."""

    @LCM_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return ""
