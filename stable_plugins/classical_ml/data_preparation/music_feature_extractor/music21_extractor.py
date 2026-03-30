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

"""music21-backed extraction helpers for MIDI and fallback analysis."""

from typing import Any

from .feature_groups import compute_feature_groups
from .payloads import empty_payload
from .utils import clip_midi_velocity, safe_float, safe_int, unique_changes


def extract_music21_payload_from_bytes(
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
            empty_payload(source_hash, fmt, source_name, warnings, part_count=None),
            0,
        )

    try:
        score = converter.parseData(data)
    except Exception as exc:
        warnings.append(f"music21 failed to parse MIDI content: {exc}")
        return (
            empty_payload(source_hash, fmt, source_name, warnings, part_count=None),
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
            duration_quarter = safe_float(
                getattr(getattr(element, "duration", None), "quarterLength", None)
            )
            if duration_quarter is not None and duration_quarter > 0:
                total_duration_quarter += duration_quarter

            if bool(getattr(element, "isRest", False)):
                if duration_quarter is not None and duration_quarter > 0:
                    rest_duration_quarter += duration_quarter
                continue

            midi_values = extract_music21_midi_values(element)
            if not midi_values:
                continue

            onset_pitches.append(midi_values[0])
            chord_sizes.append(len(midi_values))
            if duration_quarter is not None and duration_quarter > 0:
                note_durations_quarter.append(duration_quarter)

            velocity = safe_float(
                getattr(getattr(element, "volume", None), "velocity", None)
            )
            if velocity is not None:
                velocity_values.append(clip_midi_velocity(velocity))

        recurse = flat_stream.recurse() if hasattr(flat_stream, "recurse") else None
        if recurse is not None:
            for dynamic in recurse.getElementsByClass("Dynamic"):
                value = getattr(dynamic, "value", None)
                if value is not None:
                    dynamics_values.append(str(value))

            for mark in recurse.getElementsByClass("MetronomeMark"):
                bpm = safe_float(getattr(mark, "number", None))
                if bpm is None:
                    get_bpm = getattr(mark, "getQuarterBPM", None)
                    if callable(get_bpm):
                        bpm = safe_float(get_bpm())
                if bpm is None:
                    continue
                tempo_changes.append(
                    {"measure": music21_measure_hint(mark), "bpm": bpm}
                )

            for meter in recurse.getElementsByClass("TimeSignature"):
                beats = safe_int(getattr(meter, "numerator", None))
                beat_type = safe_int(getattr(meter, "denominator", None))
                if beats is None or beat_type is None:
                    ratio = str(getattr(meter, "ratioString", ""))
                    if "/" in ratio:
                        lhs, rhs = ratio.split("/", 1)
                        beats = safe_int(lhs)
                        beat_type = safe_int(rhs)
                if beats is None or beat_type is None:
                    continue
                meter_changes.append(
                    {
                        "measure": music21_measure_hint(meter),
                        "beats": beats,
                        "beat_type": beat_type,
                    }
                )

            for key in recurse.getElementsByClass("Key"):
                key_entries.append(
                    {
                        "measure": music21_measure_hint(key),
                        "fifths": safe_int(getattr(key, "sharps", None)),
                        "mode": str(getattr(key, "mode", "") or "").lower() or None,
                    }
                )

            for key_sig in recurse.getElementsByClass("KeySignature"):
                key_entries.append(
                    {
                        "measure": music21_measure_hint(key_sig),
                        "fifths": safe_int(getattr(key_sig, "sharps", None)),
                        "mode": None,
                    }
                )

        try:
            chordified = score.chordify()
            chord_recurse = (
                chordified.recurse() if hasattr(chordified, "recurse") else None
            )
            if chord_recurse is not None:
                for chord in chord_recurse.getElementsByClass("Chord"):
                    midi_values = extract_music21_midi_values(chord)
                    if not midi_values:
                        continue
                    harmony_roots.append(music21_chord_root_pc(chord, midi_values))
                    quality = getattr(chord, "quality", None) or getattr(
                        chord, "commonName", None
                    )
                    harmony_qualities.append(
                        str(quality) if quality is not None else "other"
                    )
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

    payload = empty_payload(
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

    unique_meter_changes = unique_changes(
        meter_changes,
        ("measure", "beats", "beat_type"),
    )
    unique_tempo_changes = unique_changes(tempo_changes, ("measure", "bpm"))
    payload["global"] = {
        "meter_changes": unique_meter_changes,
        "tempo_changes": unique_tempo_changes,
    }
    payload["computed"]["feature_groups"] = compute_feature_groups(
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


def try_music21_ambitus(
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


def extract_music21_midi_values(element: Any) -> list[int]:
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


def music21_chord_root_pc(chord: Any, midi_values: list[int]) -> int:
    root_method = getattr(chord, "root", None)
    if callable(root_method):
        root = root_method()
        if root is not None:
            pitch_class = getattr(root, "pitchClass", None)
            if pitch_class is not None:
                return int(pitch_class) % 12
    return midi_values[0] % 12


def music21_measure_hint(element: Any) -> int | None:
    measure = getattr(element, "measureNumber", None)
    if isinstance(measure, int):
        return measure

    offset = safe_float(getattr(element, "offset", None))
    if offset is None:
        return None
    return int(offset)
