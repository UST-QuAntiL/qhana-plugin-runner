# Copyright 2026 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from http import HTTPStatus
from typing import Mapping

from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE

from qhana_plugin_runner.api.plugin_schemas import (
    DataMetadata,
    EntryPoint,
    InputDataMetadata,
    PluginMetadata,
    PluginMetadataSchema,
    PluginType,
)
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import MusicFeatureExtractorPlugin, MusicFeatureExtractor_BLP
from .feature_extractor import MusicFeatureInput, extract_music_features
from .io_utils import load_music_sources
from .schemas import (
    InputParametersSchema,
    TaskResponseSchema,
    list_enabled_groups,
    resolve_feature_selection_from_values,
)
from .tasks import extract_music_features_task

TASK_LOGGER = get_task_logger(__name__)


@MusicFeatureExtractor_BLP.route("/")
class PluginRootView(MethodView):
    """Plugin metadata endpoint."""

    @MusicFeatureExtractor_BLP.response(HTTPStatus.OK, PluginMetadataSchema())
    @MusicFeatureExtractor_BLP.require_jwt("jwt", optional=True)
    def get(self):
        plugin = MusicFeatureExtractorPlugin.instance
        return PluginMetadata(
            title="Music Feature Extractor",
            description=plugin.description,
            name=plugin.name,
            version=plugin.version,
            type=PluginType.processing,
            entry_point=EntryPoint(
                href=url_for(f"{MusicFeatureExtractor_BLP.name}.{ProcessView.__name__}"),
                ui_href=url_for(
                    f"{MusicFeatureExtractor_BLP.name}.{MicroFrontend.__name__}"
                ),
                data_input=[
                    InputDataMetadata(
                        parameter="sourceUrl",
                        data_type="*",
                        content_type=[
                            "application/zip",
                            "application/xml",
                            "text/xml",
                            "application/vnd.recordare.musicxml+xml",
                            "audio/midi",
                            "audio/x-midi",
                            "application/octet-stream",
                        ],
                        required=True,
                    )
                ],
                data_output=[
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=["text/csv"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="entity/vector",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="entity/attribute-metadata",
                        content_type=["application/json"],
                        required=True,
                    ),
                    DataMetadata(
                        data_type="entity/list",
                        content_type=["application/json"],
                        required=True,
                    ),
                ],
            ),
            tags=plugin.tags,
        )


@MusicFeatureExtractor_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Guided micro frontend for music feature extraction."""

    @MusicFeatureExtractor_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the music feature extractor plugin."
    )
    @MusicFeatureExtractor_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @MusicFeatureExtractor_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        return self.render(request.args, errors)

    @MusicFeatureExtractor_BLP.html_response(
        HTTPStatus.OK, description="Micro frontend of the music feature extractor plugin."
    )
    @MusicFeatureExtractor_BLP.arguments(
        InputParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @MusicFeatureExtractor_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = InputParametersSchema()
        values = {
            "sourceUrl": "",
            "sourceMode": "auto",
            "declaredFormat": "auto",
            "maxFiles": 1000,
            "continueOnError": True,
            "featurePreset": "basic",
            "includePitchStats": False,
            "includeIntervals": False,
            "includeRhythm": False,
            "includeMeterTempo": False,
            "includeTexture": False,
            "includeDynamics": False,
            "includeTonality": False,
            "includeHarmony": False,
        }
        values.update(dict(data))

        preflight = None
        preflight_error = None
        if values.get("sourceUrl") and not errors:
            preflight, preflight_error = _build_preflight(values)
            if preflight_error:
                errors.setdefault("sourceUrl", []).append(preflight_error)

        return Response(
            render_template(
                "music_feature_extractor.html",
                name=MusicFeatureExtractorPlugin.instance.name,
                version=MusicFeatureExtractorPlugin.instance.version,
                schema=schema,
                values=values,
                errors=errors,
                preflight=preflight,
                process=url_for(
                    f"{MusicFeatureExtractor_BLP.name}.{ProcessView.__name__}"
                ),
            )
        )


@MusicFeatureExtractor_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long-running extraction task."""

    @MusicFeatureExtractor_BLP.arguments(
        InputParametersSchema(unknown=EXCLUDE), location="form"
    )
    @MusicFeatureExtractor_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @MusicFeatureExtractor_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        db_task = ProcessingTask(
            task_name=extract_music_features_task.name,
            parameters=InputParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        task: chain = extract_music_features_task.s(
            db_id=db_task.id
        ) | save_task_result.s(db_id=db_task.id)
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)),
            HTTPStatus.SEE_OTHER,
        )


def _build_preflight(values: Mapping[str, object]) -> tuple[dict | None, str | None]:
    source_url = str(values.get("sourceUrl") or "").strip()
    source_mode = str(values.get("sourceMode") or "auto")
    declared_format = str(values.get("declaredFormat") or "auto")
    try:
        max_files = int(values.get("maxFiles") or 1000)
    except ValueError:
        max_files = 1000
    feature_preset, feature_selection = resolve_feature_selection_from_values(values)
    enabled_groups = list_enabled_groups(feature_selection)

    try:
        sources, summary = load_music_sources(
            source_url=source_url,
            source_mode=source_mode,
            declared_format=declared_format,
            max_files=max_files,
        )
    except Exception as exc:
        return None, f"Preflight failed: {exc}"

    sample = []
    warnings: list[str] = []
    vector_ready_samples = 0
    vector_dim_count = 0
    for source in sources[:3]:
        extraction = extract_music_features(
            MusicFeatureInput(
                format=source.format,
                content=source.content,
                source_name=source.source_name,
            ),
            selection=feature_selection,
        )
        item_warnings = extraction.payload.get("warnings", [])
        warnings.extend(item_warnings)
        vector_ready = extraction.feature_vector is not None
        dim_count = (
            len(extraction.feature_vector_schema)
            if extraction.feature_vector_schema is not None
            else 0
        )
        if vector_ready:
            vector_ready_samples += 1
        if vector_dim_count == 0 and extraction.feature_vector_schema is not None:
            vector_dim_count = len(extraction.feature_vector_schema)
        sample.append(
            {
                "source_name": source.source_name,
                "format": source.format,
                "part_count": extraction.part_count,
                "vector_ready": vector_ready,
                "dim_count": dim_count,
                "warning_count": len(item_warnings),
            }
        )

    unique_warnings = list(dict.fromkeys(warnings))
    return (
        {
            "resolved_mode": summary.get("resolved_mode"),
            "source_count": summary.get("source_count"),
            "candidate_count": summary.get("candidate_count"),
            "preview_names": summary.get("preview_names", []),
            "truncated": summary.get("truncated"),
            "sample": sample,
            "vector_ready_samples": vector_ready_samples,
            "sample_count": len(sample),
            "feature_preset": feature_preset,
            "enabled_groups": enabled_groups,
            "dim_count": vector_dim_count,
            "warnings": unique_warnings[:20],
        },
        None,
    )
