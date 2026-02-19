from http import HTTPStatus
from flask import render_template, send_from_directory, redirect, request
from flask.views import MethodView
from flask.helpers import url_for
from flask.wrappers import Response
import time

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginMetadataSchema,
    EntryPoint,
    PluginType,
)

from .plugin import PA_BLP, PatternAtlas
from .pattern_atlas_dynamic.client import PatternAtlasClient, QCAtlasClient
from .pattern_atlas_dynamic.model import PatternAtlasContent, QCAtlasContent
from .pattern_atlas_dynamic.render import DynamicRender

pattern_atlas_client = PatternAtlasClient("http://localhost:1977/patternatlas")
qc_atlas_client = QCAtlasClient("http://localhost:6626/atlas")
renderer = DynamicRender()

_cache_data = None
_cache_timestamp = 0
CACHE_TTL = 24 * 60 * 60


def get_cached_atlases() -> tuple[PatternAtlasContent, QCAtlasContent]:
    global _cache_data, _cache_timestamp
    now = time.time()
    if _cache_data is None or (now - _cache_timestamp > CACHE_TTL):
        _cache_data = pattern_atlas_client.get_all(), qc_atlas_client.get_all()
        _cache_timestamp = now
    return _cache_data


def get_cached_pattern_atlas() -> PatternAtlasContent:
    return get_cached_atlases()[0]


def get_cached_qc_atlas() -> QCAtlasContent:
    return get_cached_atlases()[1]


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
                ui_href=url_for(f"{PA_BLP.name}.{IndexUI.__name__}"),
                plugin_dependencies=[],
                data_input=[],
                data_output=[],
            ),
        )


@PA_BLP.route("/ui/styles.css")
class StylesUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return send_from_directory(
            f"{PA_BLP.root_path}/pattern_atlas_dynamic/templates", "styles.css"
        )


@PA_BLP.route("/ui/graph.js")
class GraphUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return send_from_directory(
            f"{PA_BLP.root_path}/pattern_atlas_dynamic/templates", "graph.js"
        )


@PA_BLP.route("/ui/index.html")
class IndexUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        atlas = get_cached_pattern_atlas()
        html = renderer.render_index(atlas)
        return Response(html, content_type="text/html")


@PA_BLP.route("/ui/pattern-languages/<language_id>/index.html")
class LanguageUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, language_id):
        atlas = get_cached_pattern_atlas()
        language = atlas.languages.get(language_id)
        if language is None:
            return Response("Language not found", status=404)
        html = renderer.render_language_overview(atlas, language)
        return Response(html, content_type="text/html")


@PA_BLP.route("/ui/pattern-languages/<language_id>/graph.html")
class LanguageUIgraph(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, language_id):
        atlas = get_cached_pattern_atlas()
        language = atlas.languages.get(language_id)
        if language is None:
            return Response("Language not found", status=404)
        html = renderer.render_pattern_graph(atlas, language)
        return Response(html, content_type="text/html")


@PA_BLP.route("/ui/pattern-languages/<language_id>/categorized.html")
class LanguageUIcategorized(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, language_id):
        atlas = get_cached_pattern_atlas()
        language = atlas.languages.get(language_id)
        if language is None:
            return Response("Language not found", status=404)
        html = renderer.render_language_overview_categorized(atlas, language)
        return Response(html, content_type="text/html")


@PA_BLP.route("/ui/pattern-languages/<language_id>/reverse.html")
class LanguageUIreverse(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, language_id):
        atlas = get_cached_pattern_atlas()
        language = atlas.languages.get(language_id)
        if language is None:
            return Response("Language not found", status=404)
        html = renderer.render_language_overview_reverse(atlas, language)
        return Response(html, content_type="text/html")


@PA_BLP.route("/ui/pattern-languages/<language_id>/<pattern_id>/index.html")
class PatternUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, language_id, pattern_id):
        pattern_atlas = get_cached_pattern_atlas()
        qc_atlas = get_cached_qc_atlas()
        pattern = pattern_atlas.patterns.get(pattern_id)
        if pattern is None:
            return Response("Pattern not found", status=404)
        language = pattern_atlas.languages.get(language_id)
        html = renderer.render_pattern(pattern_atlas, qc_atlas, pattern, language)
        return Response(html, content_type="text/html")


@PA_BLP.route("/ui/assets/<path:asset_path>")
class AssetsUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, asset_path):
        asset_url = f"/plugins/{PA_BLP.name}/ui/assets/" + asset_path
        if asset_url in renderer._resource_bytes:
            if asset_path.endswith(".svg"):
                content_type = "image/svg+xml"
            elif asset_path.endswith(".js"):
                content_type = "application/javascript"
            elif asset_path.endswith(".woff"):
                content_type = "font/woff"
            elif asset_path.endswith(".woff2"):
                content_type = "font/woff2"
            elif asset_path.endswith(".ttf"):
                content_type = "font/ttf"
            elif asset_path.endswith(".gif"):
                content_type = "image/gif"
            else:
                content_type = "application/octet-stream"
            return Response(
                renderer._resource_bytes[asset_url], content_type=content_type
            )
        return Response("Asset not found", status=404)


@PA_BLP.route("/process/")
class Process(MethodView):
    """TODO."""

    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return ""
