from os import getenv
from http import HTTPStatus
from flask.views import MethodView
from flask.helpers import url_for
from flask.wrappers import Response
import time
from celery.utils.log import get_task_logger

from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadata,
    PluginMetadataSchema,
    EntryPoint,
    PluginType,
)
from qhana_plugin_runner.registry_client import PLUGIN_REGISTRY_CLIENT

from .plugin import PA_BLP, PatternAtlas
from .pattern_atlas_dynamic.client import PatternAtlasClient, QCAtlasClient
from .pattern_atlas_dynamic.model import PatternAtlasContent, QCAtlasContent
from .pattern_atlas_dynamic.render import DynamicRender

TASK_LOGGER = get_task_logger(__name__)

DEFAULT_CONFIG = {
    "PatternAtlasApiEndpoint": "http://localhost:1977/patternatlas",
    "QcAtlasEndpoint": "http://localhost:6626/atlas",
}


def get_config() -> dict[str, str]:
    config = dict(DEFAULT_CONFIG)
    for key, value in config.items():
        config[key] = getenv(f"PA_{key}", value)
    try:
        with PLUGIN_REGISTRY_CLIENT as client:
            services = client.fetch_by_rel(
                ["service"], {"service-id": ",".join(config.keys())}
            )
            if services is not None:
                for api_link in services.data.get("items", []):
                    service = client.fetch_by_api_link(api_link)
                    service_id = service.data.get("serviceId")
                    url = service.data.get("url")
                    if service_id is None or url is None:
                        continue
                    config[service_id] = url
    except ValueError:
        # fallback if plugin registry url is not set
        pass
    return config


renderer = DynamicRender()

_cache_data = None
_cache_timestamp = 0
CACHE_TTL = 24 * 60 * 60


def get_cached_atlases() -> tuple[PatternAtlasContent, QCAtlasContent]:
    global _cache_data, _cache_timestamp
    now = time.time()
    if _cache_data is None or (now - _cache_timestamp > CACHE_TTL):
        config = get_config()
        TASK_LOGGER.info("CONFIG", config)
        pattern_atlas_client = PatternAtlasClient(config["PatternAtlasApiEndpoint"])
        qc_atlas_client = QCAtlasClient(config["QcAtlasEndpoint"])
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
            title="Pattern Atlas",
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


@PA_BLP.route("/process/")
class Process(MethodView):
    """TODO."""

    @PA_BLP.require_jwt("jwt", optional=True)
    def get(self):
        return ""
