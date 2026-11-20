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

"""Synthetic scores and helpers shared by the music feature extractor tests."""

import base64
from fractions import Fraction
from typing import Sequence

from music21 import instrument, note, stream
from music21.midi.translate import streamToMidiFile

from music_feature_extractor.events import NoteEvent


def note_xml(
    step: str,
    octave: int,
    duration: int = 1,
    *,
    voice: str | None = None,
    chord: bool = False,
    tie: str | None = None,
) -> str:
    """Return one pitched MusicXML ``<note>`` element.

    ``tie`` is ``"start"`` or ``"stop"``.
    The duration is given in divisions.
    """
    chord_xml = "<chord/>" if chord else ""
    tie_xml = f'<tie type="{tie}"/>' if tie else ""
    voice_xml = f"<voice>{voice}</voice>" if voice else ""
    return (
        f"<note>{chord_xml}<pitch><step>{step}</step><octave>{octave}</octave></pitch>"
        f"<duration>{duration}</duration>{tie_xml}{voice_xml}</note>"
    )


def grace_note_xml(step: str, octave: int) -> str:
    """Return a MusicXML grace note, which has no duration."""
    return (
        f"<note><grace/><pitch><step>{step}</step><octave>{octave}</octave></pitch>"
        "<type>eighth</type></note>"
    )


def unpitched_note_xml(duration: int = 1) -> str:
    """Return an unpitched (percussion) MusicXML note."""
    return (
        "<note><unpitched><display-step>C</display-step>"
        f"<display-octave>5</display-octave></unpitched><duration>{duration}</duration></note>"
    )


def backup_xml(duration: int) -> str:
    """Return a ``<backup>`` element that moves back to start another voice."""
    return f"<backup><duration>{duration}</duration></backup>"


def musicxml(*measures_per_part: Sequence[str]) -> str:
    """Return a MusicXML score with one part per argument.

    Each argument lists the content of the part's measures.
    The score uses one division per quarter note and a 4/4 time signature.
    """
    part_list = "".join(
        f'<score-part id="P{number}"><part-name>Part {number}</part-name></score-part>'
        for number in range(1, len(measures_per_part) + 1)
    )
    parts = []
    for number, measures in enumerate(measures_per_part, start=1):
        measure_xml = "".join(
            f'<measure number="{index}">{_attributes(index)}{content}</measure>'
            for index, content in enumerate(measures, start=1)
        )
        parts.append(f'<part id="P{number}">{measure_xml}</part>')
    return (
        '<?xml version="1.0"?><score-partwise version="4.0">'
        f"<part-list>{part_list}</part-list>{''.join(parts)}</score-partwise>"
    )


def _attributes(measure_index: int) -> str:
    """Return the ``<attributes>`` of the first measure, "" for the others."""
    if measure_index != 1:
        return ""
    return (
        "<attributes><divisions>1</divisions>"
        "<time><beats>4</beats><beat-type>4</beat-type></time></attributes>"
    )


def midi_base64(*parts: tuple[instrument.Instrument, Sequence[str]]) -> str:
    """Return a base64 encoded MIDI file with one quarter note per given pitch.

    Each part is a ``(instrument, pitch names)`` pair and becomes its own track.
    """
    score = stream.Score()
    for part_instrument, pitch_names in parts:
        part = stream.Part()
        part.insert(0, part_instrument)
        for pitch_name in pitch_names:
            part.append(note.Note(pitch_name, quarterLength=1))
        score.insert(0, part)
    return base64.b64encode(streamToMidiFile(score).writestr()).decode()


def event(
    onset: float, offset: float, pitch: int | None, part: int = 0, voice: str = "1"
) -> NoteEvent:
    """Return a note event with positions given in quarter notes."""
    return NoteEvent(Fraction(onset), Fraction(offset), pitch, part, voice)


def monophonic_line(
    pitches: Sequence[int], part: int = 0, voice: str = "1"
) -> list[NoteEvent]:
    """Return successive quarter-note events with the given pitches."""
    return [
        event(index, index + 1, pitch, part, voice) for index, pitch in enumerate(pitches)
    ]


def nonzero_values(values: dict[str, float]) -> dict[str, float]:
    """Return the dimensions with a non-zero value, rounded to 4 places."""
    return {name: round(value, 4) for name, value in values.items() if value}
