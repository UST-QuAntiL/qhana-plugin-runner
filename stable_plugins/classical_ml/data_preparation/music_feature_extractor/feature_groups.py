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

"""Feature-group definitions and vector assembly helpers."""

import math
from typing import Any, Mapping

from .utils import clamp01, mean, safe_int, std

GROUP_ORDER = (
    "pitch_stats",
    "intervals",
    "rhythm",
    "meter_tempo",
    "texture",
    "dynamics",
    "tonality",
    "harmony",
)

BASE_FEATURE_SCHEMA = [f"pitch_class_norm_{i}" for i in range(12)] + ["ambitus_span"]
PITCH_STATS_SCHEMA = [
    "pitch_class_entropy_norm",
    "pitch_class_variety",
    "pitch_min_norm",
    "pitch_max_norm",
    "pitch_mean_norm",
    "pitch_std_norm",
    "ambitus_span_norm",
]
INTERVAL_SCHEMA = [f"mel_int_abs_norm_{i}" for i in range(13)] + [
    "mel_int_step_ratio",
    "mel_int_leap_ratio",
    "mel_int_repeat_ratio",
    "mel_int_mean_abs_norm",
]
RHYTHM_SCHEMA = [f"dur_bin_{i}" for i in range(6)] + [
    "rest_ratio",
    "note_density_norm",
    "avg_dur_norm",
]
METER_TEMPO_SCHEMA = [
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
TEXTURE_SCHEMA = [
    "part_count_norm",
    "chord_event_ratio",
    "avg_chord_size_norm",
    "chord_tone_ratio",
]
DYNAMICS_SCHEMA = [
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
TONALITY_SCHEMA = [
    "key_mode_major_ratio",
    "key_fifths_mean_norm",
    "key_change_count_norm",
]
HARMONY_SCHEMA = [f"chord_root_pc_norm_{i}" for i in range(12)] + [
    "chord_quality_major_ratio",
    "chord_quality_minor_ratio",
    "chord_quality_dim_ratio",
    "chord_quality_aug_ratio",
    "chord_quality_other_ratio",
    "harmonic_rhythm_norm",
]

GROUP_SCHEMAS = {
    "pitch_stats": PITCH_STATS_SCHEMA,
    "intervals": INTERVAL_SCHEMA,
    "rhythm": RHYTHM_SCHEMA,
    "meter_tempo": METER_TEMPO_SCHEMA,
    "texture": TEXTURE_SCHEMA,
    "dynamics": DYNAMICS_SCHEMA,
    "tonality": TONALITY_SCHEMA,
    "harmony": HARMONY_SCHEMA,
}

GROUP_WARNING_MESSAGES = {
    "pitch_stats": (
        "PitchStats enabled but no pitched notes were found; filled with 0.0 values."
    ),
    "intervals": (
        "Intervals enabled but no successive pitched onsets were found; filled with 0.0 values."
    ),
    "rhythm": "Rhythm enabled but no note durations were found; filled with 0.0 values.",
    "meter_tempo": (
        "MeterTempo enabled but no meter or tempo data was found; filled with 0.0 values."
    ),
    "texture": (
        "Texture enabled but no note onset events were found; filled with 0.0 values."
    ),
    "dynamics": (
        "Dynamics enabled but no dynamics or velocity data was found; filled with 0.0 values."
    ),
    "tonality": (
        "Tonality enabled but no key-signature data was found; filled with 0.0 values."
    ),
    "harmony": "Harmony enabled but no harmony data was found; filled with 0.0 values.",
}


def empty_feature_group_payload() -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for group in GROUP_ORDER:
        schema = GROUP_SCHEMAS[group]
        groups[group] = {
            "available": False,
            "warning": GROUP_WARNING_MESSAGES[group],
            "values": {dim: 0.0 for dim in schema},
        }
    return groups


def compute_feature_groups(
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
    groups = empty_feature_group_payload()

    available, values = compute_pitch_stats_group(pitch_values, pitch_class_counts)
    set_feature_group(groups, "pitch_stats", available=available, values=values)

    available, values = compute_intervals_group(onset_pitches)
    set_feature_group(groups, "intervals", available=available, values=values)

    available, values = compute_rhythm_group(
        note_durations_quarter=note_durations_quarter,
        total_duration_quarter=total_duration_quarter,
        rest_duration_quarter=rest_duration_quarter,
        onset_count=len(onset_pitches),
    )
    set_feature_group(groups, "rhythm", available=available, values=values)

    available, values = compute_meter_tempo_group(
        meter_changes=meter_changes,
        tempo_changes=tempo_changes,
    )
    set_feature_group(groups, "meter_tempo", available=available, values=values)

    available, values = compute_texture_group(
        part_count=part_count,
        chord_sizes=chord_sizes,
    )
    set_feature_group(groups, "texture", available=available, values=values)

    available, values = compute_dynamics_group(
        dynamics_values=dynamics_values,
        velocity_values=velocity_values,
    )
    set_feature_group(groups, "dynamics", available=available, values=values)

    available, values = compute_tonality_group(key_entries)
    set_feature_group(groups, "tonality", available=available, values=values)

    available, values, warning = compute_harmony_group(
        harmony_roots=harmony_roots,
        harmony_qualities=harmony_qualities,
        total_duration_quarter=total_duration_quarter,
        harmony_warning=harmony_warning,
    )
    set_feature_group(
        groups,
        "harmony",
        available=available,
        values=values,
        warning=warning,
    )

    return groups


def normalize_feature_selection(
    selection: Mapping[str, bool] | None,
) -> dict[str, bool]:
    normalized = {group: False for group in GROUP_ORDER}
    if selection is None:
        return normalized

    for group in GROUP_ORDER:
        normalized[group] = bool(selection.get(group, False))
    return normalized


def build_feature_vector(
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

    schema = list(BASE_FEATURE_SCHEMA)
    groups = payload.get("computed", {}).get("feature_groups", {})

    for group in GROUP_ORDER:
        if not selection.get(group, False):
            continue

        group_schema = GROUP_SCHEMAS[group]
        group_data = groups.get(group, {})
        values = group_data.get("values", {})
        vector.extend(float(values.get(name, 0.0)) for name in group_schema)
        schema.extend(group_schema)

        if not group_data.get("available", False):
            warning = group_data.get("warning") or GROUP_WARNING_MESSAGES[group]
            if warning and warning not in warnings:
                warnings.append(warning)

    return vector, schema


def set_feature_group(
    groups: dict[str, dict[str, Any]],
    group: str,
    *,
    available: bool,
    values: Mapping[str, float],
    warning: str | None = None,
) -> None:
    schema = GROUP_SCHEMAS[group]
    groups[group]["values"] = {name: float(values.get(name, 0.0)) for name in schema}
    groups[group]["available"] = bool(available)
    groups[group]["warning"] = (
        None if available else (warning or GROUP_WARNING_MESSAGES[group])
    )


def compute_pitch_stats_group(
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
    pitch_mean = mean(pitch_values)
    pitch_std = std(pitch_values)
    span = pitch_max - pitch_min

    return True, {
        "pitch_class_entropy_norm": clamp01(entropy_norm),
        "pitch_class_variety": clamp01(variety),
        "pitch_min_norm": clamp01(pitch_min / 127.0),
        "pitch_max_norm": clamp01(pitch_max / 127.0),
        "pitch_mean_norm": clamp01(pitch_mean / 127.0),
        "pitch_std_norm": clamp01(pitch_std / 64.0),
        "ambitus_span_norm": clamp01(span / 127.0),
    }


def compute_intervals_group(onset_pitches: list[int]) -> tuple[bool, dict[str, float]]:
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
    mean_abs = mean(intervals)

    values = {f"mel_int_abs_norm_{idx}": hist[idx] / count for idx in range(13)}
    values.update(
        {
            "mel_int_step_ratio": step_count / count,
            "mel_int_leap_ratio": leap_count / count,
            "mel_int_repeat_ratio": repeat_count / count,
            "mel_int_mean_abs_norm": clamp01(mean_abs / 12.0),
        }
    )
    return True, values


def compute_rhythm_group(
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
        bins[duration_bin_index(duration)] += 1

    rest_ratio = (
        rest_duration_quarter / total_duration_quarter
        if total_duration_quarter > 0
        else 0.0
    )
    note_density = (
        onset_count / total_duration_quarter if total_duration_quarter > 0 else 0.0
    )
    avg_duration = mean(note_durations_quarter)

    values = {f"dur_bin_{idx}": bins[idx] / duration_count for idx in range(6)}
    values.update(
        {
            "rest_ratio": clamp01(rest_ratio),
            "note_density_norm": clamp01(note_density / 8.0),
            "avg_dur_norm": clamp01(avg_duration / 4.0),
        }
    )
    return True, values


def compute_meter_tempo_group(
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
                "tempo_mean_norm": clamp01(mean(tempo_values) / 240.0),
                "tempo_std_norm": clamp01(std(tempo_values) / 120.0),
                "tempo_min_norm": clamp01(min(tempo_values) / 240.0),
                "tempo_max_norm": clamp01(max(tempo_values) / 240.0),
                "tempo_change_count_norm": clamp01(
                    max(len(tempo_values) - 1, 0) / 16.0
                ),
            }
        )

    meter_change_count = max(len(meter_changes) - 1, 0)
    values["meter_change_count_norm"] = clamp01(meter_change_count / 16.0)

    first_meter = meter_changes[0] if meter_changes else None
    if first_meter is not None:
        beats = safe_int(first_meter.get("beats"))
        beat_type = safe_int(first_meter.get("beat_type"))
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


def compute_texture_group(
    *,
    part_count: int | None,
    chord_sizes: list[int],
) -> tuple[bool, dict[str, float]]:
    event_count = len(chord_sizes)
    chord_event_count = sum(1 for size in chord_sizes if size > 1)
    total_tones = sum(chord_sizes)
    chord_tones = sum(size for size in chord_sizes if size > 1)

    values = {
        "part_count_norm": clamp01((part_count or 0) / 16.0),
        "chord_event_ratio": (
            chord_event_count / event_count if event_count > 0 else 0.0
        ),
        "avg_chord_size_norm": clamp01(
            (mean(chord_sizes) if event_count > 0 else 0.0) / 8.0
        ),
        "chord_tone_ratio": (chord_tones / total_tones if total_tones > 0 else 0.0),
    }
    return event_count > 0, values


def compute_dynamics_group(
    *,
    dynamics_values: list[str],
    velocity_values: list[int],
) -> tuple[bool, dict[str, float]]:
    buckets = {"pp": 0, "p": 0, "mp": 0, "mf": 0, "f": 0, "ff": 0}
    for mark in dynamics_values:
        bucket = dynamic_bucket(mark)
        if bucket is not None:
            buckets[bucket] += 1

    mark_total = sum(buckets.values())
    velocity_count = len(velocity_values)
    velocity_mean = mean(velocity_values) if velocity_values else 0.0
    velocity_std = std(velocity_values) if velocity_values else 0.0

    values = {
        "dyn_pp_ratio": (buckets["pp"] / mark_total if mark_total > 0 else 0.0),
        "dyn_p_ratio": (buckets["p"] / mark_total if mark_total > 0 else 0.0),
        "dyn_mp_ratio": (buckets["mp"] / mark_total if mark_total > 0 else 0.0),
        "dyn_mf_ratio": (buckets["mf"] / mark_total if mark_total > 0 else 0.0),
        "dyn_f_ratio": (buckets["f"] / mark_total if mark_total > 0 else 0.0),
        "dyn_ff_ratio": (buckets["ff"] / mark_total if mark_total > 0 else 0.0),
        "dyn_mark_count_norm": clamp01(mark_total / 64.0),
        "velocity_mean_norm": clamp01(velocity_mean / 127.0),
        "velocity_std_norm": clamp01(velocity_std / 64.0),
    }

    return (mark_total > 0) or (velocity_count > 0), values


def compute_tonality_group(
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

        fifths = safe_int(entry.get("fifths"))
        if fifths is not None:
            fifths_values.append(fifths)

        current = (fifths, mode)
        if previous is not None and current != previous:
            key_change_count += 1
        previous = current

    fifths_mean = mean(fifths_values) if fifths_values else 0.0

    return True, {
        "key_mode_major_ratio": major_count / key_count,
        "key_fifths_mean_norm": clamp01((fifths_mean + 7.0) / 14.0),
        "key_change_count_norm": clamp01(key_change_count / 16.0),
    }


def compute_harmony_group(
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
        quality_counts[classify_chord_quality(quality)] += 1

    values = {
        f"chord_root_pc_norm_{idx}": root_counts[idx] / event_count
        for idx in range(12)
    }
    values.update(
        {
            "chord_quality_major_ratio": quality_counts["major"] / event_count,
            "chord_quality_minor_ratio": quality_counts["minor"] / event_count,
            "chord_quality_dim_ratio": quality_counts["dim"] / event_count,
            "chord_quality_aug_ratio": quality_counts["aug"] / event_count,
            "chord_quality_other_ratio": quality_counts["other"] / event_count,
            "harmonic_rhythm_norm": clamp01(
                (event_count / total_duration_quarter) / 4.0
                if total_duration_quarter > 0
                else 0.0
            ),
        }
    )

    return True, values, None


def duration_bin_index(duration_quarter: float) -> int:
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


def dynamic_bucket(mark: str) -> str | None:
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


def classify_chord_quality(quality: str | None) -> str:
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
