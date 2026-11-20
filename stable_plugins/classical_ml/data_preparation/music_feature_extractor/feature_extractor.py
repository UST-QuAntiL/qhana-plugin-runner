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

"""Feature extraction utilities for music data (prefer MusicXML)."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .event_features import EVENT_GROUP_ORDER, compute_event_feature_groups
from .events import NoteEvent, build_note_events
from .feature_groups import build_feature_vector, normalize_feature_selection
from .music21_extractor import (
    extract_music21_payload_from_bytes,
    parse_musicxml_score,
)
from .musicxml_extractor import extract_musicxml_payload, load_musicxml_text
from .payloads import SCHEMA_VERSION, empty_payload
from .utils import sha256_hex


@dataclass(frozen=True)
class MusicFeatureInput:
    format: str
    content: str
    source_name: str | None = None


@dataclass(frozen=True)
class MusicFeatureExtraction:
    """Container for extracted music features and storage metadata."""

    source_hash: str
    format: str
    schema_version: str
    payload: dict[str, Any]
    feature_vector: list[float] | None
    feature_vector_schema: list[str] | None
    part_count: int | None
    duration_seconds: float | None


def extract_music_features(
    node: MusicFeatureInput,
    selection: Mapping[str, bool] | None = None,
) -> MusicFeatureExtraction:
    """Extract structured features from music content."""

    warnings: list[str] = []
    fmt = node.format.lower() if node.format else "unknown"
    effective_selection = normalize_feature_selection(selection)
    xml_text, source_bytes = load_musicxml_text(node.content, fmt, warnings)
    source_hash = sha256_hex(source_bytes)

    if xml_text is None:
        if fmt == "midi":
            payload, note_count, score = extract_music21_payload_from_bytes(
                source_bytes, source_hash, fmt, node.source_name, warnings
            )
            _add_event_feature_groups(
                payload, effective_selection, lambda: score, warnings
            )
            feature_vector, feature_vector_schema = build_feature_vector(
                payload,
                note_count,
                effective_selection,
                warnings,
            )
            payload["feature_vector"] = feature_vector
            payload["feature_vector_schema"] = feature_vector_schema
            part_count = payload.get("source", {}).get("partCount")
            return MusicFeatureExtraction(
                source_hash=source_hash,
                format=fmt,
                schema_version=SCHEMA_VERSION,
                payload=payload,
                feature_vector=feature_vector,
                feature_vector_schema=feature_vector_schema,
                part_count=part_count,
                duration_seconds=None,
            )

        warnings.append(f"Unsupported or unreadable format: {fmt}")
        payload = empty_payload(
            source_hash, fmt, node.source_name, warnings, part_count=None
        )
        return MusicFeatureExtraction(
            source_hash=source_hash,
            format=fmt,
            schema_version=SCHEMA_VERSION,
            payload=payload,
            feature_vector=None,
            feature_vector_schema=None,
            part_count=None,
            duration_seconds=None,
        )

    payload, note_count = extract_musicxml_payload(
        xml_text, source_hash, fmt, node.source_name, warnings
    )
    _add_event_feature_groups(
        payload, effective_selection, lambda: parse_musicxml_score(xml_text), warnings
    )

    feature_vector, feature_vector_schema = build_feature_vector(
        payload,
        note_count,
        effective_selection,
        warnings,
    )
    payload["feature_vector"] = feature_vector
    payload["feature_vector_schema"] = feature_vector_schema

    part_count = payload.get("source", {}).get("partCount")
    return MusicFeatureExtraction(
        source_hash=source_hash,
        format=fmt,
        schema_version=SCHEMA_VERSION,
        payload=payload,
        feature_vector=feature_vector,
        feature_vector_schema=feature_vector_schema,
        part_count=part_count,
        duration_seconds=None,
    )


def _add_event_feature_groups(
    payload: dict[str, Any],
    selection: Mapping[str, bool],
    load_score: Callable[[], Any],
    warnings: list[str],
) -> None:
    """Add the selected note-event feature groups to the payload.

    The score is only loaded and the note-event table only built,
    if at least one note-event group is selected.
    Otherwise the parser-based extraction keeps its output and speed.
    """
    groups = [group for group in EVENT_GROUP_ORDER if selection.get(group, False)]
    if not groups:
        return

    events = _load_note_events(load_score, warnings)
    payload["computed"]["feature_groups"].update(
        compute_event_feature_groups(events, groups)
    )


def _load_note_events(
    load_score: Callable[[], Any], warnings: list[str]
) -> list[NoteEvent]:
    """Return the note events of the score, or none if it cannot be analysed.

    A failure only makes the note-event groups unavailable.
    The parser-based groups that were already computed stay valid.
    """
    try:
        score = load_score()
        if score is None:
            # the MIDI path already reported why there is no score
            return []
        return build_note_events(score)
    except Exception as exc:  # music21 raises many unrelated exception types
        warnings.append(
            "Note-event feature groups unavailable, "
            f"music21 could not analyse the score: {exc}"
        )
        return []
