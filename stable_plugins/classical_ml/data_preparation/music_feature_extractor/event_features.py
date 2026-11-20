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

"""Feature groups computed from the note-event table.

The features follow the definitions of the jSymbolic feature catalogue
(https://jmir.sourceforge.net/manuals/jSymbolic_manual/home.html).

A melodic line is one (part, voice) pair instead of a MIDI track/channel.
If several notes of a line start together, the highest one is used.
An independent voice is a music21 part instead of a MIDI channel.
"""

import math
from collections import Counter, defaultdict
from fractions import Fraction
from functools import cached_property
from itertools import pairwise
from typing import Any, Callable, Iterable, Mapping, Sequence

from .events import NoteEvent

EVENT_GROUP_ORDER = (
    "melodic_intervals",
    "melodic_summary",
    "voice_activity",
)

LARGEST_SEPARATE_INTERVAL = 12  # an octave

MELODIC_INTERVALS_SCHEMA = [
    f"mel_m01_interval_{size:02d}" for size in range(LARGEST_SEPARATE_INTERVAL + 1)
] + [f"mel_m01_interval_{LARGEST_SEPARATE_INTERVAL + 1:02d}_plus"]
MELODIC_SUMMARY_SCHEMA = [
    "mel_m03_mean_interval",
    "mel_m09_repeated_notes",
    "mel_m10_chromatic_motion",
    "mel_m11_stepwise_motion",
    "mel_m12_melodic_thirds",
    "mel_m19_large_intervals",
    "mel_m22_rising_motion",
]
VOICE_ACTIVITY_SCHEMA = [
    "tex_t01_max_independent_voices",
    "tex_t02_avg_independent_voices",
    "tex_t03_independent_voices_std",
]

EVENT_GROUP_SCHEMAS = {
    "melodic_intervals": MELODIC_INTERVALS_SCHEMA,
    "melodic_summary": MELODIC_SUMMARY_SCHEMA,
    "voice_activity": VOICE_ACTIVITY_SCHEMA,
}

EVENT_GROUP_WARNING_MESSAGES = {
    "melodic_intervals": (
        "MelodicIntervals enabled but no successive pitched notes were found "
        "within a part or voice, filled with 0.0 values."
    ),
    "melodic_summary": (
        "MelodicSummary enabled but no successive pitched notes were found "
        "within a part or voice, filled with 0.0 values."
    ),
    "voice_activity": (
        "VoiceActivity enabled but no sounding notes were found, filled with 0.0 values."
    ),
}


class _EventAnalysis:
    """Lazily derived views of one note-event table.

    Several groups can share a derived view, e.g. the melodic intervals.
    Each view is therefore computed at most once per table.
    """

    def __init__(self, events: Sequence[NoteEvent]) -> None:
        self.events = events

    @cached_property
    def melodic_intervals(self) -> list[int]:
        """Return the signed intervals of all melodic lines."""
        return signed_melodic_intervals(self.events)


_GROUP_REDUCERS: Mapping[
    str, Callable[[_EventAnalysis], tuple[bool, dict[str, float]]]
] = {
    "melodic_intervals": lambda analysis: compute_melodic_intervals_group(
        analysis.melodic_intervals
    ),
    "melodic_summary": lambda analysis: compute_melodic_summary_group(
        analysis.melodic_intervals
    ),
    "voice_activity": lambda analysis: compute_voice_activity_group(analysis.events),
}


def compute_event_feature_groups(
    events: Sequence[NoteEvent], groups: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Return the payload entries of the requested note-event feature groups.

    Each entry has the same keys as the entries of the parser-based groups.
    ``available`` tells whether the group could be computed.
    ``warning`` is ``None`` for an available group.
    ``values`` holds one value per schema dimension, ``0.0`` if unavailable.

    Raises:
        KeyError: if a group is not a note-event feature group.
    """
    analysis = _EventAnalysis(events)
    payload: dict[str, dict[str, Any]] = {}
    for group in groups:
        available, values = _GROUP_REDUCERS[group](analysis)
        payload[group] = {
            "available": available,
            "warning": None if available else EVENT_GROUP_WARNING_MESSAGES[group],
            "values": {
                name: float(values.get(name, 0.0)) for name in EVENT_GROUP_SCHEMAS[group]
            },
        }
    return payload


def melodic_lines(events: Iterable[NoteEvent]) -> list[list[int]]:
    """Return the pitch sequence of every (part, voice) line, ordered by onset.

    Unpitched events are ignored.
    If several notes of a line start at the same onset (a chord),
    the highest pitch represents the line at that onset.
    """
    lines: dict[tuple[int, str], dict[Fraction, int]] = defaultdict(dict)
    for event in events:
        if event.pitch is None:
            continue
        line = lines[(event.part, event.voice)]
        current = line.get(event.onset)
        if current is None or event.pitch > current:
            line[event.onset] = event.pitch
    return [[line[onset] for onset in sorted(line)] for _, line in sorted(lines.items())]


def signed_melodic_intervals(events: Iterable[NoteEvent]) -> list[int]:
    """Return the signed semitone steps between successive notes of every line.

    Intervals are never formed across parts or voices.
    Rests inside a line are bridged, like in jSymbolic.
    """
    return [
        following - previous
        for line in melodic_lines(events)
        for previous, following in pairwise(line)
    ]


def compute_melodic_intervals_group(
    intervals: Sequence[int],
) -> tuple[bool, dict[str, float]]:
    """Return the compact melodic interval histogram (jSymbolic M-1).

    Bins 0 to 12 hold the share of intervals of exactly that many semitones.
    The last bin holds the share of intervals larger than an octave (M-19).
    Rising and falling intervals are counted alike.
    """
    if not intervals:
        return False, {}

    counts = Counter(min(abs(step), LARGEST_SEPARATE_INTERVAL + 1) for step in intervals)
    total = len(intervals)
    return True, {
        name: counts[size] / total for size, name in enumerate(MELODIC_INTERVALS_SCHEMA)
    }


def compute_melodic_summary_group(
    intervals: Sequence[int],
) -> tuple[bool, dict[str, float]]:
    """Return the melodic summary values of the jSymbolic M features.

    The mean interval (M-3) is given in semitones.
    All other values are shares of all melodic intervals.
    Rising motion (M-22) is the share of rising intervals among moving ones.
    Repeated notes do not count as moving intervals.
    """
    if not intervals:
        return False, {}

    sizes = Counter(abs(step) for step in intervals)
    total = len(intervals)
    size_sum = sum(size * count for size, count in sizes.items())
    large_count = sum(
        count for size, count in sizes.items() if size > LARGEST_SEPARATE_INTERVAL
    )
    rising = sum(1 for step in intervals if step > 0)
    moving = sum(1 for step in intervals if step != 0)
    return True, {
        "mel_m03_mean_interval": size_sum / total,
        "mel_m09_repeated_notes": sizes[0] / total,
        "mel_m10_chromatic_motion": sizes[1] / total,
        "mel_m11_stepwise_motion": (sizes[1] + sizes[2]) / total,
        "mel_m12_melodic_thirds": (sizes[3] + sizes[4]) / total,
        "mel_m19_large_intervals": large_count / total,
        # jSymbolic sets M-22 to 0 if there are no rising or falling intervals
        "mel_m22_rising_motion": rising / moving if moving else 0.0,
    }


def compute_voice_activity_group(
    events: Iterable[NoteEvent],
) -> tuple[bool, dict[str, float]]:
    """Return the voice activity values (jSymbolic T-1 to T-3).

    The values describe the number of parts sounding at the same time.
    T-1 is its maximum and T-2 its duration-weighted mean.
    T-3 is its duration-weighted (population) standard deviation.
    Time in which no part sounds is excluded.
    Several simultaneous notes of one part count as one voice.
    Unpitched parts count as voices, too.
    """
    segments = active_voice_segments(events)
    if not segments:
        return False, {}

    total_duration = sum(duration for duration, _ in segments)
    mean = sum(duration * voices for duration, voices in segments) / total_duration
    squared_deviations = sum(
        duration * (voices - mean) ** 2 for duration, voices in segments
    )
    variance = squared_deviations / total_duration
    return True, {
        "tex_t01_max_independent_voices": float(max(voices for _, voices in segments)),
        "tex_t02_avg_independent_voices": float(mean),
        "tex_t03_independent_voices_std": math.sqrt(variance),
    }


def active_voice_segments(events: Iterable[NoteEvent]) -> list[tuple[Fraction, int]]:
    """Return ``(duration, sounding parts)`` for segments with sounding parts.

    The segments are the intervals between successive note boundaries.
    Notes are half-open intervals ``[onset, offset)``.
    A note ending where another one starts therefore does not overlap with it.
    """
    changes: dict[Fraction, Counter[int]] = defaultdict(Counter)
    for event in events:
        changes[event.onset][event.part] += 1
        changes[event.offset][event.part] -= 1

    sounding_notes: Counter[int] = Counter()
    segments: list[tuple[Fraction, int]] = []
    for time, next_time in pairwise(sorted(changes)):
        # Counter.update adds the (possibly negative) changes per part
        sounding_notes.update(changes[time])
        sounding_parts = sum(1 for count in sounding_notes.values() if count > 0)
        if sounding_parts:
            segments.append((next_time - time, sounding_parts))
    return segments
