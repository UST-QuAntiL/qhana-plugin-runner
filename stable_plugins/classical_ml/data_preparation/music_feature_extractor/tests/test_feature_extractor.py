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

"""Tests for adding the note-event feature groups to the extraction."""

from typing import Any

from music21 import instrument

from music_feature_extractor import feature_extractor
from music_feature_extractor.event_features import EVENT_GROUP_ORDER
from music_feature_extractor.feature_extractor import (
    MusicFeatureExtraction,
    MusicFeatureInput,
    extract_music_features,
)
from music_feature_extractor.feature_groups import GROUP_ORDER

from .data import midi_base64, musicxml, nonzero_values, note_xml

LEGACY_SELECTION = {group: True for group in GROUP_ORDER}
EVENT_SELECTION = {group: True for group in EVENT_GROUP_ORDER}

TWO_STEPWISE_PARTS = musicxml(
    [note_xml("C", 4) + note_xml("D", 4) + note_xml("E", 4) + note_xml("F", 4)],
    [note_xml("C", 5) + note_xml("D", 5) + note_xml("E", 5) + note_xml("F", 5)],
)

# C-D-E-F has two whole tones and one semitone
C_TO_F_HISTOGRAM = {"mel_m01_interval_01": 0.3333, "mel_m01_interval_02": 0.6667}


def _extract(
    content: str, selection: dict, fmt: str = "musicxml"
) -> MusicFeatureExtraction:
    """Run the extraction for one source with the given group selection."""
    return extract_music_features(
        MusicFeatureInput(format=fmt, content=content, source_name="test"),
        selection=selection,
    )


def _group(extraction: MusicFeatureExtraction, group: str) -> dict[str, Any]:
    """Return the payload entry of one feature group."""
    return extraction.payload["computed"]["feature_groups"][group]


def _fail_to_parse(_xml_text: str):
    """Fail like the MusicXML parser does for a malformed score."""
    raise ValueError("cannot parse")


def test_event_groups_absent_when_not_selected():
    """Keep the payload of legacy selections free of note-event groups."""
    extraction = _extract(TWO_STEPWISE_PARTS, LEGACY_SELECTION)

    groups = set(extraction.payload["computed"]["feature_groups"])

    assert groups == set(GROUP_ORDER), (
        "Unselected note-event groups must not appear in the payload. "
        "Otherwise the raw output of existing workflows would change."
    )


def test_score_is_not_parsed_when_no_event_group_is_selected(monkeypatch):
    """Skip the music21 parse if no note-event group is selected."""
    monkeypatch.setattr(feature_extractor, "parse_musicxml_score", _fail_to_parse)

    extraction = _extract(TWO_STEPWISE_PARTS, LEGACY_SELECTION)

    warnings = extraction.payload["warnings"]
    assert not any("Note-event" in warning for warning in warnings), (
        "The parser fails in this test, so a warning means that it was called. "
        "The note-event table must only be built when one of its groups is selected."
    )


def test_event_groups_do_not_change_legacy_dimensions():
    """Append the note-event dimensions after the unchanged legacy ones."""
    legacy = _extract(TWO_STEPWISE_PARTS, LEGACY_SELECTION)
    combined = _extract(TWO_STEPWISE_PARTS, {**LEGACY_SELECTION, **EVENT_SELECTION})

    legacy_part = combined.feature_vector[: len(legacy.feature_vector)]

    assert legacy_part == legacy.feature_vector, (
        "Selecting note-event groups must not move or change any legacy dimension. "
        "Their dimensions are appended after the legacy ones."
    )


def test_musicxml_parts_are_separate_melodic_lines():
    """Compute the interval histogram per part for MusicXML."""
    extraction = _extract(TWO_STEPWISE_PARTS, {"melodic_intervals": True})

    values = _group(extraction, "melodic_intervals")["values"]

    assert nonzero_values(values) == C_TO_F_HISTOGRAM, (
        "The legacy group adds a 7 semitone interval from F4 to C5 at the part boundary. "
        "The note-event group must not."
    )


def test_midi_tracks_are_separate_melodic_lines():
    """Compute the interval histogram per track for MIDI."""
    midi = midi_base64(
        (instrument.Violin(), ["C5", "D5", "E5", "F5"]),
        (instrument.Violoncello(), ["C3", "D3", "E3", "F3"]),
    )

    extraction = _extract(midi, {"melodic_intervals": True}, fmt="midi")

    values = _group(extraction, "melodic_intervals")["values"]

    assert nonzero_values(values) == C_TO_F_HISTOGRAM, (
        "For this file the legacy group puts 100% into its octave-plus bin, "
        "because it pairs violin and cello notes. "
        "The note-event group must pair notes within a track."
    )


def test_voice_activity_is_part_of_the_vector():
    """Add the voice activity dimensions to the vector when selected."""
    extraction = _extract(TWO_STEPWISE_PARTS, {"voice_activity": True})

    dimensions = dict(zip(extraction.feature_vector_schema, extraction.feature_vector))

    assert dimensions["tex_t02_avg_independent_voices"] == 2.0, (
        "Both parts sound for the whole bar. "
        "On average two voices are therefore active."
    )


def test_parse_failure_keeps_the_legacy_vector(monkeypatch):
    """Keep the parser-based vector if music21 cannot analyse the score."""
    monkeypatch.setattr(feature_extractor, "parse_musicxml_score", _fail_to_parse)

    extraction = _extract(
        TWO_STEPWISE_PARTS, {"pitch_stats": True, "melodic_intervals": True}
    )

    assert extraction.feature_vector is not None, (
        "A music21 failure must only disable the note-event groups. "
        "The parser-based vector of the file must still be produced."
    )


def test_parse_failure_marks_event_groups_unavailable(monkeypatch):
    """Mark the note-event groups unavailable if music21 fails."""
    monkeypatch.setattr(feature_extractor, "parse_musicxml_score", _fail_to_parse)

    extraction = _extract(TWO_STEPWISE_PARTS, {"melodic_intervals": True})

    assert not _group(extraction, "melodic_intervals")["available"], (
        "Without a score there are no note events. "
        "The 0.0 values of the group must be flagged as unavailable."
    )
