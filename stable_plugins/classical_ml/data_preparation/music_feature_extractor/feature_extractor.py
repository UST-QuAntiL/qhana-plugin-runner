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

from __future__ import annotations

import base64
import hashlib
import io
import math
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping
from xml.etree import ElementTree as ET

SCHEMA_VERSION = "music-features/v1"

_STEP_TO_SEMITONE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_GROUP_ORDER = (
    "pitch_stats",
    "intervals",
    "rhythm",
    "meter_tempo",
    "texture",
    "dynamics",
    "tonality",
    "harmony",
)

_BASE_FEATURE_SCHEMA = [f"pitch_class_norm_{i}" for i in range(12)] + ["ambitus_span"]
_PITCH_STATS_SCHEMA = [
    "pitch_class_entropy_norm",
    "pitch_class_variety",
    "pitch_min_norm",
    "pitch_max_norm",
    "pitch_mean_norm",
    "pitch_std_norm",
    "ambitus_span_norm",
]
_INTERVAL_SCHEMA = [f"mel_int_abs_norm_{i}" for i in range(13)] + [
    "mel_int_step_ratio",
    "mel_int_leap_ratio",
    "mel_int_repeat_ratio",
    "mel_int_mean_abs_norm",
]
_RHYTHM_SCHEMA = [f"dur_bin_{i}" for i in range(6)] + [
    "rest_ratio",
    "note_density_norm",
    "avg_dur_norm",
]
_METER_TEMPO_SCHEMA = [
    "tempo_mean_norm",
    "tempo_std_norm",
    "tempo_min_norm",
    "tempo_max_norm",
    "tempo_change_count_norm",
    "meter_change_count_norm",
    "meter_first_2_4",
    "meter_first_3_4",
    "meter_first_4_4",
    "meter_first_6_8",
    "meter_first_other",
]
_TEXTURE_SCHEMA = [
    "part_count_norm",
    "chord_event_ratio",
    "avg_chord_size_norm",
    "chord_tone_ratio",
]
_DYNAMICS_SCHEMA = [
    "dyn_pp_ratio",
    "dyn_p_ratio",
    "dyn_mp_ratio",
    "dyn_mf_ratio",
    "dyn_f_ratio",
    "dyn_ff_ratio",
    "dyn_mark_count_norm",
    "velocity_mean_norm",
    "velocity_std_norm",
]
_TONALITY_SCHEMA = [
    "key_mode_major_ratio",
    "key_fifths_mean_norm",
    "key_change_count_norm",
]
_HARMONY_SCHEMA = [f"chord_root_pc_norm_{i}" for i in range(12)] + [
    "chord_quality_major_ratio",
    "chord_quality_minor_ratio",
    "chord_quality_dim_ratio",
    "chord_quality_aug_ratio",
    "chord_quality_other_ratio",
    "harmonic_rhythm_norm",
]

_GROUP_SCHEMAS = {
    "pitch_stats": _PITCH_STATS_SCHEMA,
    "intervals": _INTERVAL_SCHEMA,
    "rhythm": _RHYTHM_SCHEMA,
    "meter_tempo": _METER_TEMPO_SCHEMA,
    "texture": _TEXTURE_SCHEMA,
    "dynamics": _DYNAMICS_SCHEMA,
    "tonality": _TONALITY_SCHEMA,
    "harmony": _HARMONY_SCHEMA,
}

_GROUP_WARNING_MESSAGES = {
    "pitch_stats": "PitchStats enabled but no pitched notes were found; filled with 0.0 values.",
    "intervals": "Intervals enabled but no successive pitched onsets were found; filled with 0.0 values.",
    "rhythm": "Rhythm enabled but no note durations were found; filled with 0.0 values.",
    "meter_tempo": "MeterTempo enabled but no meter or tempo data was found; filled with 0.0 values.",
    "texture": "Texture enabled but no note onset events were found; filled with 0.0 values.",
    "dynamics": "Dynamics enabled but no dynamics or velocity data was found; filled with 0.0 values.",
    "tonality": "Tonality enabled but no key-signature data was found; filled with 0.0 values.",
    "harmony": "Harmony enabled but no harmony data was found; filled with 0.0 values.",
}


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
    effective_selection = _normalize_feature_selection(selection)
    xml_text, source_bytes = _load_musicxml_text(node.content, fmt, warnings)
    source_hash = _sha256_hex(source_bytes)

    if xml_text is None:
        if fmt == "midi":
            payload, note_count = _extract_music21_payload_from_bytes(
                source_bytes, source_hash, fmt, node.source_name, warnings
            )
            feature_vector, feature_vector_schema = _build_feature_vector(
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
        payload = _empty_payload(
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

    payload, note_count = _extract_musicxml_payload(
        xml_text, source_hash, fmt, node.source_name, warnings
    )

    feature_vector, feature_vector_schema = _build_feature_vector(
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


def _empty_payload(
    source_hash: str,
    fmt: str,
    source_name: str | None,
    warnings: list[str],
    *,
    part_count: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "format": fmt,
            "sourceName": source_name,
            "hash": source_hash,
            "partCount": part_count,
        },
        "per_part": [],
        "global": {"meter_changes": [], "tempo_changes": []},
        "computed": {
            "ambitus": {"lowest": None, "highest": None, "semitone_span": None},
            "pitch_class_distribution": {
                "counts": {str(i): 0 for i in range(12)},
                "normalized": {str(i): 0.0 for i in range(12)},
            },
            "feature_groups": _empty_feature_group_payload(),
        },
        "feature_vector": None,
        "feature_vector_schema": None,
        "warnings": warnings,
    }


def _empty_feature_group_payload() -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for group in _GROUP_ORDER:
        schema = _GROUP_SCHEMAS[group]
        groups[group] = {
            "available": False,
            "warning": _GROUP_WARNING_MESSAGES[group],
            "values": {dim: 0.0 for dim in schema},
        }
    return groups


def _load_musicxml_text(
    content: str | None,
    fmt: str,
    warnings: list[str],
) -> tuple[str | None, bytes]:
    """Return MusicXML text and raw source bytes for hashing."""

    if content is None:
        warnings.append("Missing content")
        return None, b""

    stripped = content.lstrip()
    if fmt in {"musicxml", "musicxml-xml", "xml"} or stripped.startswith("<"):
        source_bytes = content.encode("utf-8", errors="replace")
        return content, source_bytes

    if fmt in {"musicxml-mxl", "mxl"}:
        try:
            decoded = base64.b64decode(content)
            xml_text = _extract_xml_from_mxl(decoded, warnings)
            return xml_text, decoded
        except Exception as exc:  # pragma: no cover - defensive
            warnings.append(f"Failed to decode MXL content: {exc}")
            source_bytes = content.encode("utf-8", errors="replace")
            return None, source_bytes

    if fmt == "midi":
        try:
            decoded = base64.b64decode(content)
            return None, decoded
        except Exception as exc:
            warnings.append(f"Failed to decode MIDI content: {exc}")
            return None, content.encode("utf-8", errors="replace")

    return None, content.encode("utf-8", errors="replace")


def _extract_xml_from_mxl(data: bytes, warnings: list[str]) -> str | None:
    """Extract the MusicXML payload from an MXL zip archive."""

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "META-INF/container.xml" in archive.namelist():
                container_xml = archive.read("META-INF/container.xml")
                root_path = _parse_container_rootfile(container_xml)
                if root_path and root_path in archive.namelist():
                    return archive.read(root_path).decode("utf-8", errors="replace")

            for name in archive.namelist():
                if name.lower().endswith((".musicxml", ".xml")) and (
                    "meta-inf" not in name.lower()
                ):
                    return archive.read(name).decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        warnings.append("Invalid MXL archive")
    return None


def _parse_container_rootfile(container_xml: bytes) -> str | None:
    try:
        root = ET.fromstring(container_xml)
    except ET.ParseError:
        return None

    for rootfile in root.findall(".//{*}rootfile"):
        full_path = rootfile.get("full-path")
        if full_path:
            return full_path
    return None


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_musicxml_payload(
    xml_text: str,
    source_hash: str,
    fmt: str,
    source_name: str | None,
    warnings: list[str],
) -> tuple[dict[str, Any], int]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        warnings.append(f"Failed to parse MusicXML: {exc}")
        payload = _empty_payload(source_hash, fmt, source_name, warnings, part_count=0)
        return payload, 0

    part_list = _parse_part_list(root)
    per_part: list[dict[str, Any]] = []
    pitch_class_counts = {i: 0 for i in range(12)}
    pitch_values: list[int] = []
    onset_pitches: list[int] = []
    note_durations_quarter: list[float] = []
    total_duration_quarter = 0.0
    rest_duration_quarter = 0.0
    chord_sizes: list[int] = []
    dynamics_values: list[str] = []
    velocity_values: list[int] = []
    key_entries: list[dict[str, Any]] = []
    harmony_roots: list[int] = []
    harmony_qualities: list[str] = []

    min_midi: int | None = None
    max_midi: int | None = None
    note_count = 0
    global_time_changes: list[dict[str, Any]] = []
    global_tempo_changes: list[dict[str, Any]] = []

    for part in root.findall(".//{*}part"):
        part_id = part.get("id") or f"part-{len(per_part) + 1}"
        instrument = part_list.get(part_id)
        part_data = {
            "part_id": part_id,
            "instrument": instrument,
            "key_signatures": [],
            "time_signatures": [],
            "clefs": [],
            "tempi": [],
            "dynamics": [],
            "directions": [],
        }

        current_divisions = 1

        for measure in part.findall("./{*}measure"):
            measure_number = _parse_measure_number(measure.get("number"))

            for attributes in measure.findall("./{*}attributes"):
                divisions = _parse_int(_get_text(attributes.find("./{*}divisions")))
                if divisions is not None and divisions > 0:
                    current_divisions = divisions

                _collect_key_signatures(
                    attributes,
                    part_data["key_signatures"],
                    measure_number,
                )
                _collect_time_signatures(
                    attributes,
                    part_data["time_signatures"],
                    measure_number,
                )
                _collect_clefs(attributes, part_data["clefs"], measure_number)

            _collect_directions(measure, part_data, measure_number)

            for harmony in measure.findall("./{*}harmony"):
                root_pc, quality = _parse_harmony_event(harmony)
                if root_pc is None:
                    continue
                harmony_roots.append(root_pc)
                harmony_qualities.append(quality or "other")

            for note in measure.findall("./{*}note"):
                duration_quarter = _note_duration_quarter(note, current_divisions)
                if duration_quarter is not None and duration_quarter > 0:
                    total_duration_quarter += duration_quarter
                    if note.find("./{*}rest") is not None:
                        rest_duration_quarter += duration_quarter

                midi, pc = _pitch_to_midi_and_class(note)
                if midi is None or pc is None:
                    continue

                note_count += 1
                pitch_values.append(midi)
                pitch_class_counts[pc] += 1
                min_midi = midi if min_midi is None else min(min_midi, midi)
                max_midi = midi if max_midi is None else max(max_midi, midi)

                is_chord_continuation = note.find("./{*}chord") is not None
                if not is_chord_continuation or not chord_sizes:
                    chord_sizes.append(1)
                    onset_pitches.append(midi)
                    if duration_quarter is not None and duration_quarter > 0:
                        note_durations_quarter.append(duration_quarter)
                else:
                    chord_sizes[-1] += 1

        per_part.append(part_data)
        global_time_changes.extend(part_data["time_signatures"])
        global_tempo_changes.extend(part_data["tempi"])
        dynamics_values.extend(
            str(item.get("value") or "") for item in part_data["dynamics"]
        )
        key_entries.extend(part_data["key_signatures"])

    if min_midi is None or max_midi is None:
        alt_ambitus = _try_music21_ambitus(xml_text, warnings)
        if alt_ambitus is not None:
            min_midi, max_midi = alt_ambitus

    global_payload = {
        "meter_changes": _unique_changes(
            global_time_changes,
            ("measure", "beats", "beat_type"),
        ),
        "tempo_changes": _unique_changes(global_tempo_changes, ("measure", "bpm")),
    }

    ambitus = {
        "lowest": min_midi,
        "highest": max_midi,
        "semitone_span": (
            None if min_midi is None or max_midi is None else max_midi - min_midi
        ),
    }

    total = sum(pitch_class_counts.values())
    if total == 0:
        warnings.append("No pitched notes found for pitch class distribution")
    normalized = {
        str(pc): (pitch_class_counts[pc] / total if total > 0 else 0.0)
        for pc in range(12)
    }
    counts = {str(pc): pitch_class_counts[pc] for pc in range(12)}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "format": fmt,
            "sourceName": source_name,
            "hash": source_hash,
            "partCount": len(per_part),
        },
        "per_part": per_part,
        "global": global_payload,
        "computed": {
            "ambitus": ambitus,
            "pitch_class_distribution": {
                "counts": counts,
                "normalized": normalized,
            },
            "feature_groups": _compute_feature_groups(
                pitch_values=pitch_values,
                pitch_class_counts=pitch_class_counts,
                onset_pitches=onset_pitches,
                note_durations_quarter=note_durations_quarter,
                total_duration_quarter=total_duration_quarter,
                rest_duration_quarter=rest_duration_quarter,
                meter_changes=global_payload["meter_changes"],
                tempo_changes=global_payload["tempo_changes"],
                part_count=len(per_part),
                chord_sizes=chord_sizes,
                dynamics_values=dynamics_values,
                velocity_values=velocity_values,
                key_entries=key_entries,
                harmony_roots=harmony_roots,
                harmony_qualities=harmony_qualities,
                harmony_warning=None,
            ),
        },
        "feature_vector": None,
        "feature_vector_schema": None,
        "warnings": warnings,
    }
    return payload, note_count


def _extract_music21_payload_from_bytes(
    data: bytes,
    source_hash: str,
    fmt: str,
    source_name: str | None,
    warnings: list[str],
) -> tuple[dict[str, Any], int]:
    try:
        from music21 import converter  # type: ignore[import-not-found]
    except Exception:
        warnings.append("music21 not available for MIDI parsing")
        return (
            _empty_payload(source_hash, fmt, source_name, warnings, part_count=None),
            0,
        )

    try:
        score = converter.parseData(data)
    except Exception as exc:
        warnings.append(f"music21 failed to parse MIDI content: {exc}")
        return (
            _empty_payload(source_hash, fmt, source_name, warnings, part_count=None),
            0,
        )

    pitch_values = [
        int(midi)
        for midi in (
            getattr(pitch, "midi", None)
            for pitch in (getattr(score, "pitches", None) or [])
        )
        if midi is not None
    ]
    pitch_class_counts = {i: 0 for i in range(12)}
    for midi in pitch_values:
        pitch_class_counts[midi % 12] += 1

    onset_pitches: list[int] = []
    note_durations_quarter: list[float] = []
    total_duration_quarter = 0.0
    rest_duration_quarter = 0.0
    chord_sizes: list[int] = []
    dynamics_values: list[str] = []
    velocity_values: list[int] = []
    key_entries: list[dict[str, Any]] = []
    tempo_changes: list[dict[str, Any]] = []
    meter_changes: list[dict[str, Any]] = []
    harmony_roots: list[int] = []
    harmony_qualities: list[str] = []
    harmony_warning: str | None = None

    flat_stream = None
    try:
        flat_stream = score.flatten()
    except Exception:
        flat_stream = None

    if flat_stream is not None:
        for element in getattr(flat_stream, "notesAndRests", []):
            duration_quarter = _safe_float(
                getattr(getattr(element, "duration", None), "quarterLength", None)
            )
            if duration_quarter is not None and duration_quarter > 0:
                total_duration_quarter += duration_quarter

            if bool(getattr(element, "isRest", False)):
                if duration_quarter is not None and duration_quarter > 0:
                    rest_duration_quarter += duration_quarter
                continue

            midi_values = _extract_music21_midi_values(element)
            if not midi_values:
                continue

            onset_pitches.append(midi_values[0])
            chord_sizes.append(len(midi_values))
            if duration_quarter is not None and duration_quarter > 0:
                note_durations_quarter.append(duration_quarter)

            velocity = _safe_float(
                getattr(getattr(element, "volume", None), "velocity", None)
            )
            if velocity is not None:
                velocity_values.append(_clip_midi_velocity(velocity))

        recurse = flat_stream.recurse() if hasattr(flat_stream, "recurse") else None
        if recurse is not None:
            for dynamic in recurse.getElementsByClass("Dynamic"):
                value = getattr(dynamic, "value", None)
                if value is not None:
                    dynamics_values.append(str(value))

            for mark in recurse.getElementsByClass("MetronomeMark"):
                bpm = _safe_float(getattr(mark, "number", None))
                if bpm is None:
                    get_bpm = getattr(mark, "getQuarterBPM", None)
                    if callable(get_bpm):
                        bpm = _safe_float(get_bpm())
                if bpm is None:
                    continue
                tempo_changes.append(
                    {"measure": _music21_measure_hint(mark), "bpm": bpm}
                )

            for meter in recurse.getElementsByClass("TimeSignature"):
                beats = _safe_int(getattr(meter, "numerator", None))
                beat_type = _safe_int(getattr(meter, "denominator", None))
                if beats is None or beat_type is None:
                    ratio = str(getattr(meter, "ratioString", ""))
                    if "/" in ratio:
                        lhs, rhs = ratio.split("/", 1)
                        beats = _safe_int(lhs)
                        beat_type = _safe_int(rhs)
                if beats is None or beat_type is None:
                    continue
                meter_changes.append(
                    {
                        "measure": _music21_measure_hint(meter),
                        "beats": beats,
                        "beat_type": beat_type,
                    }
                )

            for key in recurse.getElementsByClass("Key"):
                key_entries.append(
                    {
                        "measure": _music21_measure_hint(key),
                        "fifths": _safe_int(getattr(key, "sharps", None)),
                        "mode": str(getattr(key, "mode", "") or "").lower() or None,
                    }
                )

            for key_sig in recurse.getElementsByClass("KeySignature"):
                key_entries.append(
                    {
                        "measure": _music21_measure_hint(key_sig),
                        "fifths": _safe_int(getattr(key_sig, "sharps", None)),
                        "mode": None,
                    }
                )

        try:
            chordified = score.chordify()
            chord_recurse = chordified.recurse() if hasattr(chordified, "recurse") else None
            if chord_recurse is not None:
                for chord in chord_recurse.getElementsByClass("Chord"):
                    midi_values = _extract_music21_midi_values(chord)
                    if not midi_values:
                        continue
                    harmony_roots.append(_music21_chord_root_pc(chord, midi_values))
                    quality = getattr(chord, "quality", None) or getattr(
                        chord,
                        "commonName",
                        None,
                    )
                    harmony_qualities.append(str(quality) if quality is not None else "other")
        except Exception as exc:
            harmony_warning = f"Harmony analysis with music21 failed: {exc}"

    total = len(pitch_values)
    if total == 0:
        warnings.append("No pitched notes found in MIDI data")

    normalized = {
        str(pc): (pitch_class_counts[pc] / total if total > 0 else 0.0)
        for pc in range(12)
    }
    counts = {str(pc): pitch_class_counts[pc] for pc in range(12)}

    min_midi = min(pitch_values) if pitch_values else None
    max_midi = max(pitch_values) if pitch_values else None

    part_count = None
    try:
        parts = getattr(score, "parts", None)
        if parts is not None:
            part_count = len(parts)
    except Exception:
        part_count = None

    payload = _empty_payload(
        source_hash,
        fmt,
        source_name,
        warnings,
        part_count=part_count,
    )
    payload["computed"]["ambitus"] = {
        "lowest": min_midi,
        "highest": max_midi,
        "semitone_span": (
            None if min_midi is None or max_midi is None else max_midi - min_midi
        ),
    }
    payload["computed"]["pitch_class_distribution"] = {
        "counts": counts,
        "normalized": normalized,
    }

    unique_meter_changes = _unique_changes(
        meter_changes,
        ("measure", "beats", "beat_type"),
    )
    unique_tempo_changes = _unique_changes(tempo_changes, ("measure", "bpm"))
    payload["global"] = {
        "meter_changes": unique_meter_changes,
        "tempo_changes": unique_tempo_changes,
    }
    payload["computed"]["feature_groups"] = _compute_feature_groups(
        pitch_values=pitch_values,
        pitch_class_counts=pitch_class_counts,
        onset_pitches=onset_pitches,
        note_durations_quarter=note_durations_quarter,
        total_duration_quarter=total_duration_quarter,
        rest_duration_quarter=rest_duration_quarter,
        meter_changes=unique_meter_changes,
        tempo_changes=unique_tempo_changes,
        part_count=part_count,
        chord_sizes=chord_sizes,
        dynamics_values=dynamics_values,
        velocity_values=velocity_values,
        key_entries=key_entries,
        harmony_roots=harmony_roots,
        harmony_qualities=harmony_qualities,
        harmony_warning=harmony_warning,
    )
    return payload, total


def _compute_feature_groups(
    *,
    pitch_values: list[int],
    pitch_class_counts: dict[int, int],
    onset_pitches: list[int],
    note_durations_quarter: list[float],
    total_duration_quarter: float,
    rest_duration_quarter: float,
    meter_changes: list[dict[str, Any]],
    tempo_changes: list[dict[str, Any]],
    part_count: int | None,
    chord_sizes: list[int],
    dynamics_values: list[str],
    velocity_values: list[int],
    key_entries: list[dict[str, Any]],
    harmony_roots: list[int],
    harmony_qualities: list[str],
    harmony_warning: str | None,
) -> dict[str, dict[str, Any]]:
    groups = _empty_feature_group_payload()

    available, values = _compute_pitch_stats_group(pitch_values, pitch_class_counts)
    _set_feature_group(groups, "pitch_stats", available=available, values=values)

    available, values = _compute_intervals_group(onset_pitches)
    _set_feature_group(groups, "intervals", available=available, values=values)

    available, values = _compute_rhythm_group(
        note_durations_quarter=note_durations_quarter,
        total_duration_quarter=total_duration_quarter,
        rest_duration_quarter=rest_duration_quarter,
        onset_count=len(onset_pitches),
    )
    _set_feature_group(groups, "rhythm", available=available, values=values)

    available, values = _compute_meter_tempo_group(
        meter_changes=meter_changes,
        tempo_changes=tempo_changes,
    )
    _set_feature_group(groups, "meter_tempo", available=available, values=values)

    available, values = _compute_texture_group(part_count=part_count, chord_sizes=chord_sizes)
    _set_feature_group(groups, "texture", available=available, values=values)

    available, values = _compute_dynamics_group(
        dynamics_values=dynamics_values,
        velocity_values=velocity_values,
    )
    _set_feature_group(groups, "dynamics", available=available, values=values)

    available, values = _compute_tonality_group(key_entries)
    _set_feature_group(groups, "tonality", available=available, values=values)

    available, values, warning = _compute_harmony_group(
        harmony_roots=harmony_roots,
        harmony_qualities=harmony_qualities,
        total_duration_quarter=total_duration_quarter,
        harmony_warning=harmony_warning,
    )
    _set_feature_group(
        groups,
        "harmony",
        available=available,
        values=values,
        warning=warning,
    )

    return groups


def _set_feature_group(
    groups: dict[str, dict[str, Any]],
    group: str,
    *,
    available: bool,
    values: Mapping[str, float],
    warning: str | None = None,
) -> None:
    schema = _GROUP_SCHEMAS[group]
    groups[group]["values"] = {name: float(values.get(name, 0.0)) for name in schema}
    groups[group]["available"] = bool(available)
    groups[group]["warning"] = None if available else (warning or _GROUP_WARNING_MESSAGES[group])


def _compute_pitch_stats_group(
    pitch_values: list[int],
    pitch_class_counts: dict[int, int],
) -> tuple[bool, dict[str, float]]:
    total = len(pitch_values)
    if total == 0:
        return False, {}

    probs = [
        pitch_class_counts.get(pc, 0) / total
        for pc in range(12)
        if pitch_class_counts.get(pc, 0) > 0
    ]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    entropy_norm = entropy / math.log(12) if probs else 0.0
    variety = len(probs) / 12.0

    pitch_min = min(pitch_values)
    pitch_max = max(pitch_values)
    pitch_mean = _mean(pitch_values)
    pitch_std = _std(pitch_values)
    span = pitch_max - pitch_min

    return True, {
        "pitch_class_entropy_norm": _clamp01(entropy_norm),
        "pitch_class_variety": _clamp01(variety),
        "pitch_min_norm": _clamp01(pitch_min / 127.0),
        "pitch_max_norm": _clamp01(pitch_max / 127.0),
        "pitch_mean_norm": _clamp01(pitch_mean / 127.0),
        "pitch_std_norm": _clamp01(pitch_std / 64.0),
        "ambitus_span_norm": _clamp01(span / 127.0),
    }


def _compute_intervals_group(onset_pitches: list[int]) -> tuple[bool, dict[str, float]]:
    intervals = [
        abs(onset_pitches[index + 1] - onset_pitches[index])
        for index in range(len(onset_pitches) - 1)
    ]
    count = len(intervals)
    if count == 0:
        return False, {}

    hist = [0 for _ in range(13)]
    for value in intervals:
        hist[min(value, 12)] += 1

    step_count = sum(1 for value in intervals if 1 <= value <= 2)
    leap_count = sum(1 for value in intervals if value > 2)
    repeat_count = sum(1 for value in intervals if value == 0)
    mean_abs = _mean(intervals)

    values = {f"mel_int_abs_norm_{idx}": hist[idx] / count for idx in range(13)}
    values.update(
        {
            "mel_int_step_ratio": step_count / count,
            "mel_int_leap_ratio": leap_count / count,
            "mel_int_repeat_ratio": repeat_count / count,
            "mel_int_mean_abs_norm": _clamp01(mean_abs / 12.0),
        }
    )
    return True, values


def _compute_rhythm_group(
    *,
    note_durations_quarter: list[float],
    total_duration_quarter: float,
    rest_duration_quarter: float,
    onset_count: int,
) -> tuple[bool, dict[str, float]]:
    duration_count = len(note_durations_quarter)
    if duration_count == 0:
        return False, {}

    bins = [0 for _ in range(6)]
    for duration in note_durations_quarter:
        bins[_duration_bin_index(duration)] += 1

    rest_ratio = (
        rest_duration_quarter / total_duration_quarter if total_duration_quarter > 0 else 0.0
    )
    note_density = onset_count / total_duration_quarter if total_duration_quarter > 0 else 0.0
    avg_duration = _mean(note_durations_quarter)

    values = {f"dur_bin_{idx}": bins[idx] / duration_count for idx in range(6)}
    values.update(
        {
            "rest_ratio": _clamp01(rest_ratio),
            "note_density_norm": _clamp01(note_density / 8.0),
            "avg_dur_norm": _clamp01(avg_duration / 4.0),
        }
    )
    return True, values


def _compute_meter_tempo_group(
    *,
    meter_changes: list[dict[str, Any]],
    tempo_changes: list[dict[str, Any]],
) -> tuple[bool, dict[str, float]]:
    tempo_values = [
        float(item["bpm"])
        for item in tempo_changes
        if isinstance(item.get("bpm"), (int, float))
    ]

    values = {
        "tempo_mean_norm": 0.0,
        "tempo_std_norm": 0.0,
        "tempo_min_norm": 0.0,
        "tempo_max_norm": 0.0,
        "tempo_change_count_norm": 0.0,
        "meter_change_count_norm": 0.0,
        "meter_first_2_4": 0.0,
        "meter_first_3_4": 0.0,
        "meter_first_4_4": 0.0,
        "meter_first_6_8": 0.0,
        "meter_first_other": 0.0,
    }

    if tempo_values:
        values.update(
            {
                "tempo_mean_norm": _clamp01(_mean(tempo_values) / 240.0),
                "tempo_std_norm": _clamp01(_std(tempo_values) / 120.0),
                "tempo_min_norm": _clamp01(min(tempo_values) / 240.0),
                "tempo_max_norm": _clamp01(max(tempo_values) / 240.0),
                "tempo_change_count_norm": _clamp01(
                    max(len(tempo_values) - 1, 0) / 16.0
                ),
            }
        )

    meter_change_count = max(len(meter_changes) - 1, 0)
    values["meter_change_count_norm"] = _clamp01(meter_change_count / 16.0)

    first_meter = meter_changes[0] if meter_changes else None
    if first_meter is not None:
        beats = _safe_int(first_meter.get("beats"))
        beat_type = _safe_int(first_meter.get("beat_type"))
        if (beats, beat_type) == (2, 4):
            values["meter_first_2_4"] = 1.0
        elif (beats, beat_type) == (3, 4):
            values["meter_first_3_4"] = 1.0
        elif (beats, beat_type) == (4, 4):
            values["meter_first_4_4"] = 1.0
        elif (beats, beat_type) == (6, 8):
            values["meter_first_6_8"] = 1.0
        else:
            values["meter_first_other"] = 1.0

    available = bool(tempo_values or meter_changes)
    return available, values


def _compute_texture_group(
    *,
    part_count: int | None,
    chord_sizes: list[int],
) -> tuple[bool, dict[str, float]]:
    event_count = len(chord_sizes)
    chord_event_count = sum(1 for size in chord_sizes if size > 1)
    total_tones = sum(chord_sizes)
    chord_tones = sum(size for size in chord_sizes if size > 1)

    values = {
        "part_count_norm": _clamp01((part_count or 0) / 16.0),
        "chord_event_ratio": (
            chord_event_count / event_count if event_count > 0 else 0.0
        ),
        "avg_chord_size_norm": _clamp01(
            (_mean(chord_sizes) if event_count > 0 else 0.0) / 8.0
        ),
        "chord_tone_ratio": (chord_tones / total_tones if total_tones > 0 else 0.0),
    }
    return event_count > 0, values


def _compute_dynamics_group(
    *,
    dynamics_values: list[str],
    velocity_values: list[int],
) -> tuple[bool, dict[str, float]]:
    buckets = {"pp": 0, "p": 0, "mp": 0, "mf": 0, "f": 0, "ff": 0}
    for mark in dynamics_values:
        bucket = _dynamic_bucket(mark)
        if bucket is not None:
            buckets[bucket] += 1

    mark_total = sum(buckets.values())
    velocity_count = len(velocity_values)
    velocity_mean = _mean(velocity_values) if velocity_values else 0.0
    velocity_std = _std(velocity_values) if velocity_values else 0.0

    values = {
        "dyn_pp_ratio": (buckets["pp"] / mark_total if mark_total > 0 else 0.0),
        "dyn_p_ratio": (buckets["p"] / mark_total if mark_total > 0 else 0.0),
        "dyn_mp_ratio": (buckets["mp"] / mark_total if mark_total > 0 else 0.0),
        "dyn_mf_ratio": (buckets["mf"] / mark_total if mark_total > 0 else 0.0),
        "dyn_f_ratio": (buckets["f"] / mark_total if mark_total > 0 else 0.0),
        "dyn_ff_ratio": (buckets["ff"] / mark_total if mark_total > 0 else 0.0),
        "dyn_mark_count_norm": _clamp01(mark_total / 64.0),
        "velocity_mean_norm": _clamp01(velocity_mean / 127.0),
        "velocity_std_norm": _clamp01(velocity_std / 64.0),
    }

    return (mark_total > 0) or (velocity_count > 0), values


def _compute_tonality_group(
    key_entries: list[dict[str, Any]],
) -> tuple[bool, dict[str, float]]:
    key_count = len(key_entries)
    if key_count == 0:
        return False, {}

    major_count = 0
    fifths_values: list[int] = []
    key_change_count = 0
    previous: tuple[int | None, str | None] | None = None

    for entry in key_entries:
        mode_raw = entry.get("mode")
        mode = str(mode_raw).strip().lower() if mode_raw is not None else None
        if mode == "major":
            major_count += 1

        fifths = _safe_int(entry.get("fifths"))
        if fifths is not None:
            fifths_values.append(fifths)

        current = (fifths, mode)
        if previous is not None and current != previous:
            key_change_count += 1
        previous = current

    fifths_mean = _mean(fifths_values) if fifths_values else 0.0

    return True, {
        "key_mode_major_ratio": major_count / key_count,
        "key_fifths_mean_norm": _clamp01((fifths_mean + 7.0) / 14.0),
        "key_change_count_norm": _clamp01(key_change_count / 16.0),
    }


def _compute_harmony_group(
    *,
    harmony_roots: list[int],
    harmony_qualities: list[str],
    total_duration_quarter: float,
    harmony_warning: str | None,
) -> tuple[bool, dict[str, float], str | None]:
    event_count = len(harmony_roots)
    if event_count == 0:
        return False, {}, harmony_warning

    root_counts = [0 for _ in range(12)]
    for root in harmony_roots:
        root_counts[root % 12] += 1

    quality_counts = {
        "major": 0,
        "minor": 0,
        "dim": 0,
        "aug": 0,
        "other": 0,
    }
    for quality in harmony_qualities:
        quality_counts[_classify_chord_quality(quality)] += 1

    values = {
        f"chord_root_pc_norm_{idx}": root_counts[idx] / event_count for idx in range(12)
    }
    values.update(
        {
            "chord_quality_major_ratio": quality_counts["major"] / event_count,
            "chord_quality_minor_ratio": quality_counts["minor"] / event_count,
            "chord_quality_dim_ratio": quality_counts["dim"] / event_count,
            "chord_quality_aug_ratio": quality_counts["aug"] / event_count,
            "chord_quality_other_ratio": quality_counts["other"] / event_count,
            "harmonic_rhythm_norm": _clamp01(
                (event_count / total_duration_quarter) / 4.0
                if total_duration_quarter > 0
                else 0.0
            ),
        }
    )

    return True, values, None


def _normalize_feature_selection(
    selection: Mapping[str, bool] | None,
) -> dict[str, bool]:
    normalized = {group: False for group in _GROUP_ORDER}
    if selection is None:
        return normalized

    for group in _GROUP_ORDER:
        normalized[group] = bool(selection.get(group, False))
    return normalized


def _build_feature_vector(
    payload: dict[str, Any],
    note_count: int,
    selection: Mapping[str, bool],
    warnings: list[str],
) -> tuple[list[float] | None, list[str] | None]:
    if note_count == 0:
        return None, None

    distribution = payload.get("computed", {}).get("pitch_class_distribution", {})
    normalized_distribution = distribution.get("normalized", {})
    vector = [float(normalized_distribution.get(str(i), 0.0)) for i in range(12)]

    ambitus = payload.get("computed", {}).get("ambitus", {})
    span = ambitus.get("semitone_span")
    vector.append(float(span) if span is not None else 0.0)

    schema = list(_BASE_FEATURE_SCHEMA)
    groups = payload.get("computed", {}).get("feature_groups", {})

    for group in _GROUP_ORDER:
        if not selection.get(group, False):
            continue

        group_schema = _GROUP_SCHEMAS[group]
        group_data = groups.get(group, {})
        values = group_data.get("values", {})
        vector.extend(float(values.get(name, 0.0)) for name in group_schema)
        schema.extend(group_schema)

        if not group_data.get("available", False):
            warning = group_data.get("warning") or _GROUP_WARNING_MESSAGES[group]
            if warning and warning not in warnings:
                warnings.append(warning)

    return vector, schema


def _try_music21_ambitus(
    xml_text: str,
    warnings: list[str],
) -> tuple[int | None, int | None] | None:
    try:
        from music21 import converter  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        score = converter.parseData(xml_text)
        pitches = [pitch.midi for pitch in score.pitches]
        if not pitches:
            return None
        return min(pitches), max(pitches)
    except Exception as exc:
        warnings.append(f"music21 ambitus fallback failed: {exc}")
        return None


def _parse_part_list(root: ET.Element) -> dict[str, str | None]:
    part_map: dict[str, str | None] = {}
    for score_part in root.findall(".//{*}part-list/{*}score-part"):
        part_id = score_part.get("id")
        if part_id is None:
            continue
        part_name = _get_text(score_part.find("./{*}part-name"))
        part_map[part_id] = part_name
    return part_map


def _collect_key_signatures(
    attributes: ET.Element,
    bucket: list[dict[str, Any]],
    measure_number: int | str | None,
) -> None:
    for key in attributes.findall("./{*}key"):
        fifths = _parse_int(_get_text(key.find("./{*}fifths")))
        mode = _get_text(key.find("./{*}mode"))
        if fifths is None and mode is None:
            continue
        bucket.append(
            {
                "measure": measure_number,
                "fifths": fifths,
                "mode": mode,
            }
        )


def _collect_time_signatures(
    attributes: ET.Element,
    bucket: list[dict[str, Any]],
    measure_number: int | str | None,
) -> None:
    for time in attributes.findall("./{*}time"):
        beats = _parse_int(_get_text(time.find("./{*}beats")))
        beat_type = _parse_int(_get_text(time.find("./{*}beat-type")))
        if beats is None or beat_type is None:
            continue
        bucket.append(
            {
                "measure": measure_number,
                "beats": beats,
                "beat_type": beat_type,
            }
        )


def _collect_clefs(
    attributes: ET.Element,
    bucket: list[dict[str, Any]],
    measure_number: int | str | None,
) -> None:
    for clef in attributes.findall("./{*}clef"):
        sign = _get_text(clef.find("./{*}sign"))
        line = _parse_int(_get_text(clef.find("./{*}line")))
        if sign is None and line is None:
            continue
        bucket.append({"measure": measure_number, "sign": sign, "line": line})


def _collect_directions(
    measure: ET.Element,
    part_data: dict[str, Any],
    measure_number: int | str | None,
) -> None:
    for direction in measure.findall("./{*}direction"):
        for direction_type in direction.findall("./{*}direction-type"):
            for dynamics in direction_type.findall("./{*}dynamics"):
                for child in list(dynamics):
                    value = _strip_tag(child.tag)
                    if value:
                        part_data["dynamics"].append(
                            {"measure": measure_number, "value": value}
                        )
            for words in direction_type.findall("./{*}words"):
                text = _get_text(words)
                if text:
                    part_data["directions"].append(
                        {"measure": measure_number, "text": text}
                    )

        sound = direction.find("./{*}sound")
        if sound is not None and "tempo" in sound.attrib:
            bpm = _parse_float(sound.attrib.get("tempo"))
            if bpm is not None:
                part_data["tempi"].append({"measure": measure_number, "bpm": bpm})


def _parse_harmony_event(harmony: ET.Element) -> tuple[int | None, str | None]:
    root = harmony.find("./{*}root")
    step = _get_text(root.find("./{*}root-step")) if root is not None else None
    alter = (
        _parse_int(_get_text(root.find("./{*}root-alter"))) if root is not None else None
    )

    root_pc: int | None = None
    if step is not None:
        semitone = _STEP_TO_SEMITONE.get(step.upper())
        if semitone is not None:
            root_pc = (semitone + (alter or 0)) % 12

    kind_element = harmony.find("./{*}kind")
    kind = _get_text(kind_element)
    if kind is None and kind_element is not None:
        kind = kind_element.get("text")

    return root_pc, kind


def _note_duration_quarter(note: ET.Element, divisions: int) -> float | None:
    if divisions <= 0:
        return None
    duration = _parse_float(_get_text(note.find("./{*}duration")))
    if duration is None or duration < 0:
        return None
    return float(duration) / float(divisions)


def _pitch_to_midi_and_class(note: ET.Element) -> tuple[int | None, int | None]:
    if note.find("./{*}rest") is not None:
        return None, None

    pitch = note.find("./{*}pitch")
    if pitch is None:
        return None, None

    step = _get_text(pitch.find("./{*}step"))
    octave = _parse_int(_get_text(pitch.find("./{*}octave")))
    if step is None or octave is None:
        return None, None

    semitone = _STEP_TO_SEMITONE.get(step.upper())
    if semitone is None:
        return None, None

    alter = _parse_int(_get_text(pitch.find("./{*}alter"))) or 0
    midi = (octave + 1) * 12 + semitone + alter
    pitch_class = (semitone + alter) % 12
    return midi, pitch_class


def _extract_music21_midi_values(element: Any) -> list[int]:
    if bool(getattr(element, "isChord", False)):
        values: list[int] = []
        for pitch in getattr(element, "pitches", []) or []:
            midi = getattr(pitch, "midi", None)
            if midi is not None:
                values.append(int(midi))
        return values

    pitch = getattr(element, "pitch", None)
    midi = getattr(pitch, "midi", None) if pitch is not None else None
    if midi is None:
        return []
    return [int(midi)]


def _music21_chord_root_pc(chord: Any, midi_values: list[int]) -> int:
    root_method = getattr(chord, "root", None)
    if callable(root_method):
        root = root_method()
        if root is not None:
            pitch_class = getattr(root, "pitchClass", None)
            if pitch_class is not None:
                return int(pitch_class) % 12
    return midi_values[0] % 12


def _music21_measure_hint(element: Any) -> int | None:
    measure = getattr(element, "measureNumber", None)
    if isinstance(measure, int):
        return measure

    offset = _safe_float(getattr(element, "offset", None))
    if offset is None:
        return None
    return int(offset)


def _duration_bin_index(duration_quarter: float) -> int:
    if duration_quarter <= 0.25:
        return 0
    if duration_quarter <= 0.5:
        return 1
    if duration_quarter <= 1.0:
        return 2
    if duration_quarter <= 2.0:
        return 3
    if duration_quarter <= 4.0:
        return 4
    return 5


def _dynamic_bucket(mark: str) -> str | None:
    normalized = mark.strip().lower()
    if normalized in {"pp", "ppp", "pppp"}:
        return "pp"
    if normalized == "p":
        return "p"
    if normalized == "mp":
        return "mp"
    if normalized == "mf":
        return "mf"
    if normalized == "f":
        return "f"
    if normalized in {"ff", "fff", "ffff"}:
        return "ff"
    return None


def _classify_chord_quality(quality: str | None) -> str:
    normalized = (quality or "").strip().lower()
    if "dim" in normalized:
        return "dim"
    if "aug" in normalized or "+" in normalized:
        return "aug"
    if "minor" in normalized or normalized.startswith("min"):
        return "minor"
    if "major" in normalized or "maj" in normalized:
        return "major"
    return "other"


def _mean(values: list[int] | list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _std(values: list[int] | list[float]) -> float:
    if not values:
        return 0.0
    mean_value = _mean(values)
    variance = sum((float(value) - mean_value) ** 2 for value in values) / float(
        len(values)
    )
    return math.sqrt(variance)


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _clip_midi_velocity(value: float) -> int:
    rounded = int(round(value))
    if rounded < 0:
        return 0
    if rounded > 127:
        return 127
    return rounded


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _unique_changes(
    items: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        key = tuple(item.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    unique.sort(key=_measure_sort_key)
    return unique


def _measure_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    measure = item.get("measure")
    if isinstance(measure, int):
        return (measure, "")
    if isinstance(measure, str) and measure.isdigit():
        return (int(measure), "")
    return (10**9, str(measure))


def _parse_measure_number(value: str | None) -> int | str | None:
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    return value


def _get_text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    text = element.text.strip()
    return text if text else None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _strip_tag(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag
