from http import HTTPStatus
from flask import render_template, send_from_directory, redirect
from flask.views import MethodView
from flask.helpers import url_for
from flask.wrappers import Response
from pathlib import Path

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginMetadataSchema,
    EntryPoint,
    PluginType,
)

from .plugin import PA_BLP, PatternAtlas


@PA_BLP.route("/")
class PluginRootView(MethodView):
    @PA_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return PluginMetadata(
            title="PA",
            description=PatternAtlas.instance.description,
            name=PatternAtlas.instance.name,
            version=PatternAtlas.instance.version,
            tags=PatternAtlas.instance.tags,
            type=PluginType.interaction,
            entry_point=EntryPoint(
                href=url_for(f"{PA_BLP.name}.{Process.__name__}"),
                ui_href=url_for(f"{PA_BLP.name}.{UI.__name__}", path="index.html"),
                plugin_dependencies=[],
                data_input=[],
                data_output=[],
            ),
        )


@PA_BLP.route("/ui/<path:path>")
class UI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, path):
        # send_from_directory returns 404 Not Found when you try to
        # access a directory, thus we have to detect this and send a
        # redirect
        if (Path(PA_BLP.root_path) / "static" / path).is_dir():
            return redirect(
                url_for(f"{PA_BLP.name}.{UI.__name__}", path=f"{path}/index.html")
            )
        return send_from_directory(f"{PA_BLP.root_path}/static", path)


@PA_BLP.route("/process/")
class Process(MethodView):
    """TODO."""

    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return ""
