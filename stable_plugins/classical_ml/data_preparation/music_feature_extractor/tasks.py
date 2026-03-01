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

import re
from pathlib import PurePath
from tempfile import SpooledTemporaryFile
from typing import Optional

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.storage import STORE

from . import MusicFeatureExtractorPlugin
from .feature_extractor import MusicFeatureInput, extract_music_features
from .io_utils import LoadedMusicSource, load_music_sources
from .schemas import (
    InputParameters,
    InputParametersSchema,
    resolve_feature_selection_from_input,
)

TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(
    name=f"{MusicFeatureExtractorPlugin.instance.identifier}.extract_music_features",
    bind=True,
)
def extract_music_features_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting music feature extraction task with db id '{db_id}'.")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)
    TASK_LOGGER.info(f"Loaded input parameters from db: {input_params}")
    feature_selection = resolve_feature_selection_from_input(input_params)

    sources, source_summary = load_music_sources(
        source_url=input_params.source_url,
        source_mode=input_params.source_mode,
        declared_format=input_params.declared_format,
        max_files=input_params.max_files,
    )

    vector_rows: list[tuple[str, dict[str, float]]] = []
    vector_schema: list[str] = []
    raw_rows: list[dict] = []
    error_count = 0

    for index, source in enumerate(sources, start=1):
        entity_id = _entity_id_from_source(source, index)
        try:
            extraction = extract_music_features(
                MusicFeatureInput(
                    format=source.format,
                    content=source.content,
                    source_name=source.source_name,
                ),
                selection=feature_selection,
            )
        except Exception as exc:
            error_count += 1
            raw_rows.append(
                {
                    "ID": entity_id,
                    "href": "",
                    "status": "error",
                    "sourceName": source.source_name,
                    "format": source.format,
                    "error": str(exc),
                    "vectorReady": False,
                }
            )
            if not input_params.continue_on_error:
                raise ValueError(
                    f"Extraction failed for source '{source.source_name}': {exc}"
                ) from exc
            continue

        dim_map: dict[str, float] = {}
        if extraction.feature_vector and extraction.feature_vector_schema:
            dim_map = {
                name: float(value)
                for name, value in zip(
                    extraction.feature_vector_schema, extraction.feature_vector
                )
            }
            for dim_name in dim_map:
                if dim_name not in vector_schema:
                    vector_schema.append(dim_name)
            vector_rows.append((entity_id, dim_map))

        raw_rows.append(
            {
                "ID": entity_id,
                "href": "",
                "status": "ok",
                "sourceName": source.source_name,
                "format": extraction.format,
                "sourceHash": extraction.source_hash,
                "schemaVersion": extraction.schema_version,
                "partCount": extraction.part_count,
                "durationSeconds": extraction.duration_seconds,
                "vectorReady": bool(dim_map),
                "warnings": extraction.payload.get("warnings", []),
                "features": extraction.payload,
            }
        )

    if not vector_rows:
        raise ValueError("No vectorizable music features were extracted.")

    vector_entities = [
        {
            "ID": entity_id,
            "href": "",
            **{dim_name: dim_map.get(dim_name, 0.0) for dim_name in vector_schema},
        }
        for entity_id, dim_map in vector_rows
    ]

    vector_attributes = ["ID", "href", *vector_schema]
    metadata_entities = _build_vector_metadata(vector_schema)

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(
            vector_entities,
            output,
            "text/csv",
            attributes=vector_attributes,
        )
        STORE.persist_task_result(
            db_id,
            output,
            "music-feature-vectors.csv",
            "entity/vector",
            "text/csv",
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(vector_entities, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "music-feature-vectors.json",
            "entity/vector",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(metadata_entities, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "music-feature-vector-metadata.json",
            "entity/attribute-metadata",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(raw_rows, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "music-feature-raw.json",
            "entity/list",
            "application/json",
        )

    return (
        f"Extracted vectors for {len(vector_entities)} music sources "
        f"(errors: {error_count}, input candidates: {source_summary.get('candidate_count')})."
    )


def _entity_id_from_source(source: LoadedMusicSource, index: int) -> str:
    stem = PurePath(source.source_name).stem or f"source_{index}"
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_") or "source"
    return f"music_{index:04d}_{safe}"


def _build_vector_metadata(vector_schema: list[str]) -> list[dict]:
    metadata = [
        AttributeMetadata(
            ID="ID",
            attribute_type="string",
            title="ID",
            description="Stable entity identifier.",
        ).to_dict(),
        AttributeMetadata(
            ID="href",
            attribute_type="string",
            title="Href",
            description="Optional reference URL.",
        ).to_dict(),
    ]
    for dim in vector_schema:
        metadata.append(
            AttributeMetadata(
                ID=dim,
                attribute_type="number",
                title=dim,
                description=f"Extracted music feature '{dim}'.",
            ).to_dict()
        )
    return metadata
