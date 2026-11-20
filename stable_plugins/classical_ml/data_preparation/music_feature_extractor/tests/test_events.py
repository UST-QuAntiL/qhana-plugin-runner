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

"""Tests for building the note-event table from music21 scores."""

import base64
from fractions import Fraction

from music21 import converter, instrument

from music_feature_extractor.events import NoteEvent, build_note_events
from music_feature_extractor.music21_extractor import parse_musicxml_score

from .data import (
    backup_xml,
    grace_note_xml,
    midi_base64,
    musicxml,
    note_xml,
    unpitched_note_xml,
)


def _events(xml: str) -> list[NoteEvent]:
    """Return the note events of a MusicXML string."""
    return build_note_events(parse_musicxml_score(xml))


def test_tied_notes_become_one_event():
    """Merge a note tied across the bar line into one event."""
    first_measure = note_xml("C", 5, 3) + note_xml("D", 5, 1, tie="start")
    second_measure = note_xml("D", 5, 2, tie="stop") + note_xml("E", 5, 2)
    xml = musicxml([first_measure, second_measure])

    d_events = [e for e in _events(xml) if e.pitch == 74]

    assert [(e.onset, e.offset) for e in d_events] == [(Fraction(3), Fraction(6))], (
        "A tied note is one attack. "
        "Counting the continuation as a second note would add a fake repeated note."
    )


def test_grace_notes_are_skipped():
    """Leave out grace notes, which have no duration."""
    xml = musicxml([grace_note_xml("B", 4) + note_xml("C", 5, 4)])

    pitches = [e.pitch for e in _events(xml)]

    assert pitches == [72], (
        "Grace notes have no duration. "
        "They would create zero-length segments in the voice activity sweep."
    )


def test_chord_tones_become_separate_events():
    """Create one event per chord tone at the same position."""
    chord = "".join(
        [
            note_xml("C", 4, 4),
            note_xml("E", 4, 4, chord=True),
            note_xml("G", 4, 4, chord=True),
        ]
    )
    xml = musicxml([chord])

    events = _events(xml)

    assert [(e.onset, e.pitch) for e in events] == [(0, 60), (0, 64), (0, 67)], (
        "Every chord tone is a sounding note. "
        "Dropping tones would lose information needed by later feature groups."
    )


def test_explicit_voices_keep_their_id():
    """Keep the MusicXML voice of notes written with <backup>."""
    measure = "".join(
        [
            note_xml("C", 5, 2, voice="1"),
            note_xml("D", 5, 2, voice="1"),
            backup_xml(4),
            note_xml("C", 3, 4, voice="2"),
        ]
    )
    xml = musicxml([measure])

    voices = {e.pitch: e.voice for e in _events(xml)}

    assert voices == {72: "1", 74: "1", 48: "2"}, (
        "Without the voice id, "
        "melodic intervals would be formed between the upper and the lower voice."
    )


def test_notes_without_explicit_voice_use_default_voice():
    """Assign voice "1" when a measure has no explicit voices."""
    xml = musicxml([note_xml("C", 5, 4)])

    voices = {e.voice for e in _events(xml)}

    assert voices == {"1"}, (
        "Notes outside explicit voices must share one default voice, "
        "so that a line continues across measures with and without voices."
    )


def test_parts_get_their_index():
    """Store the index of the part each note belongs to."""
    xml = musicxml([note_xml("C", 5, 4)], [note_xml("C", 3, 4)])

    parts = {e.pitch: e.part for e in _events(xml)}

    assert parts == {72: 0, 48: 1}, (
        "The part index separates melodic lines. "
        "It also defines what counts as an independent voice."
    )


def test_unpitched_notes_have_no_pitch():
    """Represent unpitched percussion notes with pitch None."""
    xml = musicxml([unpitched_note_xml(2) + unpitched_note_xml(2)])

    pitches = [e.pitch for e in _events(xml)]

    assert pitches == [None, None], (
        "Percussion notes must stay in the table because they count as sounding. "
        "They must not be mistaken for pitches."
    )


def test_midi_tracks_become_parts():
    """Build one part per MIDI track."""
    data = midi_base64(
        (instrument.Violin(), ["C5", "D5"]), (instrument.Violoncello(), ["C3", "D3"])
    )
    score = converter.parseData(base64.b64decode(data))

    parts = {e.pitch: e.part for e in build_note_events(score)}

    assert parts == {72: 0, 74: 0, 48: 1, 50: 1}, (
        "MIDI tracks of different instruments must not be merged into one part. "
        "Otherwise melodies would be formed across instruments."
    )


def test_events_are_ordered_by_onset():
    """Order the events by onset before part."""
    xml = musicxml(
        [note_xml("C", 5, 2) + note_xml("D", 5, 2)],
        [note_xml("C", 3, 2) + note_xml("D", 3, 2)],
    )

    onsets = [e.onset for e in _events(xml)]

    assert onsets == sorted(onsets), (
        "The table is documented to be ordered by onset. "
        "Consumers may rely on that order."
    )
