from http import HTTPStatus
from flask import render_template, send_from_directory, redirect
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
from .pattern_atlas_dynamic.client import AtlasClient
from .pattern_atlas_dynamic.render import DynamicRender


atlas_client = AtlasClient("http://localhost:1977/patternatlas")
renderer = DynamicRender()

_cache_data = None
_cache_timestamp = 0
CACHE_TTL = 24 * 60 * 60

def get_cached_atlas():

    global _cache_data, _cache_timestamp
    now = time.time()
    if _cache_data is None or (now - _cache_timestamp > CACHE_TTL):
        _cache_data = atlas_client.get_all()
        _cache_timestamp = now
    return _cache_data


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
                plugin_dependencies=["httpx~=0.25.0", "jinja2~=3.1.2","mistune~=3.0.2","markupsafe~=2.1.3"],
                data_input=[],
                data_output=[],
            ),
        )


@PA_BLP.route("/ui/index.html")
class IndexUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        atlas = get_cached_atlas()
        html = renderer.render_index(atlas)
        return Response(html, content_type="text/html")


@PA_BLP.route("/ui/pattern-languages/<language_id>/index.html")
class LanguageUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, language_id):
        atlas = get_cached_atlas()
        language = atlas.languages.get(language_id)
        if language is None:
            return Response("Language not found", status=404)
        html = renderer.render_language_overview(atlas, language)
        return Response(html, content_type="text/html")


@PA_BLP.route("/ui/pattern-languages/<language_id>/<pattern_id>/index.html")
class PatternUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, language_id, pattern_id):
        atlas = get_cached_atlas()
        pattern = atlas.patterns.get(pattern_id)
        if pattern is None:
            return Response("Pattern not found", status=404)
        language = atlas.languages.get(language_id)
        html = renderer.render_pattern(atlas, pattern, language)
        return Response(html, content_type="text/html")



@PA_BLP.route("/ui/assets/<path:asset_path>")
class AssetsUI(MethodView):
    @PA_BLP.response(HTTPStatus.OK)
    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self, asset_path):
        asset_url = "/assets/" + asset_path
        if asset_url in renderer._resource_bytes:
            return Response(
                renderer._resource_bytes[asset_url],
                content_type="application/octet-stream"
            )
        return Response("Asset not found", status=404)


@PA_BLP.route("/process/")
class Process(MethodView):
    """TODO."""

    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return ""
