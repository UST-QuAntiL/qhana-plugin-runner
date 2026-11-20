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

"""Tests for the feature groups computed from the note-event table."""

import pytest

from music_feature_extractor.event_features import (
    EVENT_GROUP_SCHEMAS,
    compute_event_feature_groups,
    compute_melodic_intervals_group,
    compute_melodic_summary_group,
    compute_voice_activity_group,
    signed_melodic_intervals,
)

from .data import event, monophonic_line, nonzero_values

C4, C_SHARP4, D_SHARP4 = 60, 61, 63


def test_interval_histogram_worked_example():
    """Reproduce the M-1 worked example C4, C#4, D#4, C#4, C#4."""
    events = monophonic_line([C4, C_SHARP4, D_SHARP4, C_SHARP4, C_SHARP4])

    available, values = compute_melodic_intervals_group(signed_melodic_intervals(events))

    expected = {
        "mel_m01_interval_00": 0.25,
        "mel_m01_interval_01": 0.25,
        "mel_m01_interval_02": 0.5,
    }
    assert available, "Four successive intervals exist, so the group must be available."
    assert nonzero_values(values) == expected, (
        "The intervals [1, 2, 2, 0] must give the shares 0.25, 0.25 and 0.5. "
        "This is the M-1 worked example of the jSymbolic assessment."
    )


def test_intervals_are_not_formed_across_parts():
    """Form no interval at the boundary between two stepwise parts."""
    events = monophonic_line([60, 62, 64], part=0) + monophonic_line([72, 74, 76], part=1)

    intervals = signed_melodic_intervals(events)

    assert sorted(intervals) == [2, 2, 2, 2], (
        "The legacy intervals group adds a fake 8 semitone jump between these parts. "
        "The note-event groups must only pair notes of the same part."
    )


def test_intervals_are_not_formed_across_voices():
    """Pair notes only within one voice of a part."""
    events = monophonic_line([72, 74], voice="1") + monophonic_line([48, 50], voice="2")

    intervals = signed_melodic_intervals(events)

    assert sorted(intervals) == [2, 2], (
        "The upper and the lower voice of one part are separate melodic lines. "
        "An interval between them is not a melodic step."
    )


def test_highest_note_represents_a_chord_in_a_line():
    """Use the highest pitch when several notes of a line start together."""
    events = [event(0, 1, 60), event(0, 1, 67), event(1, 2, 69)]

    intervals = signed_melodic_intervals(events)

    assert intervals == [2], (
        "The chord [60, 67] is represented by its top note 67. "
        "The step to 69 is therefore 2 semitones."
    )


def test_unpitched_events_are_ignored_for_melody():
    """Exclude percussion events from melodic lines."""
    events = [event(0, 1, 60), event(1, 2, None), event(2, 3, 62)]

    intervals = signed_melodic_intervals(events)

    assert intervals == [2], (
        "Percussion hits have no pitch. "
        "They must not interrupt or extend a melodic line."
    )


def test_octave_and_larger_intervals_use_separate_bins():
    """Separate the octave (12) from intervals larger than an octave."""
    events = monophonic_line([60, 72, 91])

    _, values = compute_melodic_intervals_group(signed_melodic_intervals(events))

    expected = {"mel_m01_interval_12": 0.5, "mel_m01_interval_13_plus": 0.5}
    assert nonzero_values(values) == expected, (
        "An octave (12) and a 19 semitone leap must not share a bin. "
        "The legacy intervals group puts both into its last bin."
    )


def test_melodic_summary_worked_example():
    """Compute the summary values of the M-1 worked example."""
    events = monophonic_line([C4, C_SHARP4, D_SHARP4, C_SHARP4, C_SHARP4])

    _, values = compute_melodic_summary_group(signed_melodic_intervals(events))

    expected = {
        "mel_m03_mean_interval": 1.25,
        "mel_m09_repeated_notes": 0.25,
        "mel_m10_chromatic_motion": 0.25,
        "mel_m11_stepwise_motion": 0.75,
        "mel_m22_rising_motion": round(2 / 3, 4),
    }
    assert nonzero_values(values) == expected, (
        "The intervals are [+1, +2, -2, 0], so the mean interval is 1.25. "
        "Two of the three moving intervals rise."
    )


def test_rising_motion_is_zero_without_moving_intervals():
    """Follow jSymbolic and set M-22 to 0 for a line of repeated notes."""
    events = monophonic_line([60, 60, 60])

    _, values = compute_melodic_summary_group(signed_melodic_intervals(events))

    assert values["mel_m22_rising_motion"] == 0.0, (
        "Without rising or falling intervals the ratio is undefined. "
        "jSymbolic uses 0 in that case."
    )


@pytest.mark.parametrize(
    "compute", [compute_melodic_intervals_group, compute_melodic_summary_group]
)
def test_melodic_groups_unavailable_without_intervals(compute):
    """Report melodic groups as unavailable for a single note."""
    available, values = compute(signed_melodic_intervals([event(0, 1, 60)]))

    assert (available, values) == (False, {}), (
        "A single note has no interval. "
        "The group must be unavailable instead of claiming 0 values."
    )


def test_voice_activity_worked_example():
    """Weight the voice count by duration (A: 4 quarters, B: the last one)."""
    events = [event(0, 4, 72, part=0), event(3, 4, 64, part=1)]

    _, values = compute_voice_activity_group(events)

    expected = {
        "tex_t01_max_independent_voices": 2.0,
        "tex_t02_avg_independent_voices": 1.25,
        "tex_t03_independent_voices_std": 0.433,
    }
    assert nonzero_values(values) == expected, (
        "One voice sounds for 3 quarters and two voices sound for 1 quarter. "
        "This gives a mean of 1.25 and a standard deviation of 0.433."
    )


def test_touching_notes_do_not_overlap():
    """Treat a note ending where another starts as not simultaneous."""
    events = [event(0, 1, 72, part=0), event(1, 2, 64, part=1)]

    _, values = compute_voice_activity_group(events)

    assert values["tex_t01_max_independent_voices"] == 1.0, (
        "Notes are half-open intervals [onset, offset). "
        "A note ending where another one starts never overlaps with it."
    )


def test_silence_is_excluded_from_voice_activity():
    """Leave out time in which no part sounds."""
    events = [event(0, 1, 72, part=0), event(3, 4, 72, part=0)]

    _, values = compute_voice_activity_group(events)

    assert values["tex_t02_avg_independent_voices"] == 1.0, (
        "jSymbolic excludes rests from T-2. "
        "The 2 silent quarters must not lower the mean."
    )


def test_chord_in_one_part_counts_as_one_voice():
    """Count a part with simultaneous notes as one voice."""
    events = [event(0, 1, 60, part=0), event(0, 1, 64, part=0), event(0, 1, 67, part=0)]

    _, values = compute_voice_activity_group(events)

    assert values["tex_t01_max_independent_voices"] == 1.0, (
        "Independent voices are parts, like MIDI channels in jSymbolic. "
        "Chord tones of one part are not separate voices."
    )


def test_unpitched_parts_count_as_voices():
    """Count a percussion part as a sounding voice."""
    events = [event(0, 1, 60, part=0), event(0, 1, None, part=1)]

    _, values = compute_voice_activity_group(events)

    assert values["tex_t01_max_independent_voices"] == 2.0, (
        "jSymbolic counts the percussion channel as a voice. "
        "The note-event groups follow that definition."
    )


def test_voice_activity_unavailable_without_events():
    """Report voice activity as unavailable for an empty table."""
    assert compute_voice_activity_group([]) == (False, {}), (
        "An empty score has no sounding voice. " "The group must be unavailable."
    )


def test_only_requested_groups_are_computed():
    """Return payload entries only for the requested groups."""
    payload = compute_event_feature_groups(monophonic_line([60, 62]), ["voice_activity"])

    assert list(payload) == ["voice_activity"], (
        "Unrequested groups must not appear in the payload. "
        "Otherwise the raw output of existing selections would change."
    )


def test_unavailable_group_payload_has_all_dimensions():
    """Fill every schema dimension of an unavailable group with 0.0."""
    payload = compute_event_feature_groups([], ["melodic_summary"])["melodic_summary"]

    expected = dict.fromkeys(EVENT_GROUP_SCHEMAS["melodic_summary"], 0.0)
    assert payload["values"] == expected, (
        "The vector builder expects a value for every dimension of a selected group. "
        "An unavailable group therefore gets 0.0 for every dimension."
    )
