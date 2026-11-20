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

"""Note-event table built from a music21 score.

The table keeps information that the parser-based feature groups discard:
which part and voice a note belongs to and when it starts *and* ends.
It is only built when a feature group that needs it is selected.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

DEFAULT_VOICE = "1"


@dataclass(frozen=True)
class NoteEvent:
    """One sounding note of a score.

    Chords contribute one event per chord tone.
    Tied notes are merged into one event.
    Grace notes are left out.

    Attributes:
        onset: start of the note in quarter notes from the start of the score.
        offset: end of the note in quarter notes (exclusive).
        pitch: MIDI pitch number, ``None`` for unpitched percussion.
        part: index of the music21 part.
            music21 splits a piano with two staves into two parts.
        voice: id of the explicit voice inside the part.
            Notes outside an explicit voice get the voice ``"1"``.
    """

    onset: Fraction
    offset: Fraction
    pitch: int | None
    part: int
    voice: str


def build_note_events(score: Any) -> list[NoteEvent]:
    """Return the note events of all parts of a music21 score.

    Args:
        score: a parsed ``music21.stream.Score``.

    Returns:
        the events, ordered by onset, part, voice and pitch.
    """
    # music21 is imported lazily, like in music21_extractor.py,
    # so that the parser-based feature groups keep working without music21
    from music21.note import Unpitched  # type: ignore[import-not-found]
    from music21.percussion import PercussionChord  # type: ignore[import-not-found]
    from music21.stream import Voice  # type: ignore[import-not-found]

    events: list[NoteEvent] = []
    for part_index, part in enumerate(score.parts):
        merged_part = part.stripTies()
        for element in merged_part.recurse().notes:
            if element.duration.isGrace or element.quarterLength <= 0:
                continue
            onset = Fraction(element.getOffsetInHierarchy(merged_part))
            offset = onset + Fraction(element.quarterLength)
            # recurse() sets the activeSite to the stream containing the element
            site = element.activeSite
            voice = str(site.id) if isinstance(site, Voice) else DEFAULT_VOICE
            for pitch in _element_pitches(element, Unpitched, PercussionChord):
                events.append(NoteEvent(onset, offset, pitch, part_index, voice))

    events.sort(key=_event_sort_key)
    return events


def _element_pitches(
    element: Any, unpitched_type: type, percussion_chord_type: type
) -> list[int | None]:
    """Return the MIDI pitch of every note head of a note, chord or drum hit.

    Unpitched note heads map to ``None``.
    They need explicit handling because music21 reports no ``pitches`` for them,
    which would drop drum hits from the table.
    """
    if isinstance(element, unpitched_type):
        return [None]
    if isinstance(element, percussion_chord_type):
        return [
            None if isinstance(head, unpitched_type) else int(head.pitch.midi)
            for head in element.notes
        ]
    return [int(pitch.midi) for pitch in element.pitches]


def _event_sort_key(event: NoteEvent) -> tuple[Fraction, int, str, int]:
    """Return a key that orders events by onset, part, voice and pitch."""
    pitch = -1 if event.pitch is None else event.pitch
    return (event.onset, event.part, event.voice, pitch)
