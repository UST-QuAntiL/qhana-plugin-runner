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

"""MusicXML and MXL extraction helpers."""

import base64
import io
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

from .feature_groups import compute_feature_groups
from .music21_extractor import try_music21_ambitus
from .payloads import SCHEMA_VERSION, empty_payload
from .utils import (
    get_text,
    parse_float,
    parse_int,
    parse_measure_number,
    strip_tag,
    unique_changes,
)

STEP_TO_SEMITONE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


def load_musicxml_text(
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
            xml_text = extract_xml_from_mxl(decoded, warnings)
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


def extract_musicxml_payload(
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
        payload = empty_payload(source_hash, fmt, source_name, warnings, part_count=0)
        return payload, 0

    part_list = parse_part_list(root)
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
            measure_number = parse_measure_number(measure.get("number"))

            for attributes in measure.findall("./{*}attributes"):
                divisions = parse_int(get_text(attributes.find("./{*}divisions")))
                if divisions is not None and divisions > 0:
                    current_divisions = divisions

                collect_key_signatures(
                    attributes,
                    part_data["key_signatures"],
                    measure_number,
                )
                collect_time_signatures(
                    attributes,
                    part_data["time_signatures"],
                    measure_number,
                )
                collect_clefs(attributes, part_data["clefs"], measure_number)

            collect_directions(measure, part_data, measure_number)

            for harmony in measure.findall("./{*}harmony"):
                root_pc, quality = parse_harmony_event(harmony)
                if root_pc is None:
                    continue
                harmony_roots.append(root_pc)
                harmony_qualities.append(quality or "other")

            for note in measure.findall("./{*}note"):
                duration_quarter = note_duration_quarter(note, current_divisions)
                if duration_quarter is not None and duration_quarter > 0:
                    total_duration_quarter += duration_quarter
                    if note.find("./{*}rest") is not None:
                        rest_duration_quarter += duration_quarter

                midi, pc = pitch_to_midi_and_class(note)
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
        alt_ambitus = try_music21_ambitus(xml_text, warnings)
        if alt_ambitus is not None:
            min_midi, max_midi = alt_ambitus

    global_payload = {
        "meter_changes": unique_changes(
            global_time_changes,
            ("measure", "beats", "beat_type"),
        ),
        "tempo_changes": unique_changes(global_tempo_changes, ("measure", "bpm")),
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
            "feature_groups": compute_feature_groups(
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


def extract_xml_from_mxl(data: bytes, warnings: list[str]) -> str | None:
    """Extract the MusicXML payload from an MXL zip archive."""

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "META-INF/container.xml" in archive.namelist():
                container_xml = archive.read("META-INF/container.xml")
                root_path = parse_container_rootfile(container_xml)
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


def parse_container_rootfile(container_xml: bytes) -> str | None:
    try:
        root = ET.fromstring(container_xml)
    except ET.ParseError:
        return None

    for rootfile in root.findall(".//{*}rootfile"):
        full_path = rootfile.get("full-path")
        if full_path:
            return full_path
    return None


def parse_part_list(root: ET.Element) -> dict[str, str | None]:
    part_map: dict[str, str | None] = {}
    for score_part in root.findall(".//{*}part-list/{*}score-part"):
        part_id = score_part.get("id")
        if part_id is None:
            continue
        part_name = get_text(score_part.find("./{*}part-name"))
        part_map[part_id] = part_name
    return part_map


def collect_key_signatures(
    attributes: ET.Element,
    bucket: list[dict[str, Any]],
    measure_number: int | str | None,
) -> None:
    for key in attributes.findall("./{*}key"):
        fifths = parse_int(get_text(key.find("./{*}fifths")))
        mode = get_text(key.find("./{*}mode"))
        if fifths is None and mode is None:
            continue
        bucket.append(
            {
                "measure": measure_number,
                "fifths": fifths,
                "mode": mode,
            }
        )


def collect_time_signatures(
    attributes: ET.Element,
    bucket: list[dict[str, Any]],
    measure_number: int | str | None,
) -> None:
    for time in attributes.findall("./{*}time"):
        beats = parse_int(get_text(time.find("./{*}beats")))
        beat_type = parse_int(get_text(time.find("./{*}beat-type")))
        if beats is None or beat_type is None:
            continue
        bucket.append(
            {
                "measure": measure_number,
                "beats": beats,
                "beat_type": beat_type,
            }
        )


def collect_clefs(
    attributes: ET.Element,
    bucket: list[dict[str, Any]],
    measure_number: int | str | None,
) -> None:
    for clef in attributes.findall("./{*}clef"):
        sign = get_text(clef.find("./{*}sign"))
        line = parse_int(get_text(clef.find("./{*}line")))
        if sign is None and line is None:
            continue
        bucket.append({"measure": measure_number, "sign": sign, "line": line})


def collect_directions(
    measure: ET.Element,
    part_data: dict[str, Any],
    measure_number: int | str | None,
) -> None:
    for direction in measure.findall("./{*}direction"):
        for direction_type in direction.findall("./{*}direction-type"):
            for dynamics in direction_type.findall("./{*}dynamics"):
                for child in list(dynamics):
                    value = strip_tag(child.tag)
                    if value:
                        part_data["dynamics"].append(
                            {"measure": measure_number, "value": value}
                        )
            for words in direction_type.findall("./{*}words"):
                text = get_text(words)
                if text:
                    part_data["directions"].append(
                        {"measure": measure_number, "text": text}
                    )

        sound = direction.find("./{*}sound")
        if sound is not None and "tempo" in sound.attrib:
            bpm = parse_float(sound.attrib.get("tempo"))
            if bpm is not None:
                part_data["tempi"].append({"measure": measure_number, "bpm": bpm})


def parse_harmony_event(harmony: ET.Element) -> tuple[int | None, str | None]:
    root = harmony.find("./{*}root")
    step = get_text(root.find("./{*}root-step")) if root is not None else None
    alter = (
        parse_int(get_text(root.find("./{*}root-alter"))) if root is not None else None
    )

    root_pc: int | None = None
    if step is not None:
        semitone = STEP_TO_SEMITONE.get(step.upper())
        if semitone is not None:
            root_pc = (semitone + (alter or 0)) % 12

    kind_element = harmony.find("./{*}kind")
    kind = get_text(kind_element)
    if kind is None and kind_element is not None:
        kind = kind_element.get("text")

    return root_pc, kind


def note_duration_quarter(note: ET.Element, divisions: int) -> float | None:
    if divisions <= 0:
        return None
    duration = parse_float(get_text(note.find("./{*}duration")))
    if duration is None or duration < 0:
        return None
    return float(duration) / float(divisions)


def pitch_to_midi_and_class(note: ET.Element) -> tuple[int | None, int | None]:
    if note.find("./{*}rest") is not None:
        return None, None

    pitch = note.find("./{*}pitch")
    if pitch is None:
        return None, None

    step = get_text(pitch.find("./{*}step"))
    octave = parse_int(get_text(pitch.find("./{*}octave")))
    if step is None or octave is None:
        return None, None

    semitone = STEP_TO_SEMITONE.get(step.upper())
    if semitone is None:
        return None, None

    alter = parse_int(get_text(pitch.find("./{*}alter"))) or 0
    midi = (octave + 1) * 12 + semitone + alter
    pitch_class = (semitone + alter) % 12
    return midi, pitch_class
