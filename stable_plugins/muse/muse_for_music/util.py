# Copyright 2024 QHAna plugin runner contributors.
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

from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from urllib.parse import urlparse

from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata


class EntityId(NamedTuple):
    id_: str
    href: str


def entity_to_id(entity: Dict) -> EntityId:
    href: Optional[str] = None

    match entity:
        case {"_links": {"self": {"href": str(url)}}}:
            href = url
        case {"self": {"href": str(url)}}:
            href = url
        case {"href": str(url)}:
            href = url
        case str(url):
            href = url

    if href is None:
        raise ValueError("Could not extract url from entity!")

    path = urlparse(href).path.split("/")

    # clean up path (remove prefix part and empty segments)
    while path and path[0] != "api":
        path = path[1:]
    path = [s for s in path[1:] if s]

    identifier = "".join(path)

    match path:
        case ["taxonomies", "tree" | "list", str(tax)]:
            identifier = f"t_{tax}"
        case ["taxonomies", "tree" | "list", str(tax), str(item_id)]:
            if entity.get("name") == "na":
                item_id = "na"
            identifier = f"t_{tax}_{item_id}"
        case ["persons", str(person)]:
            identifier = f"c_{person}"
        case ["opuses", str(opus)]:
            identifier = f"o_{opus}"
        case ["parts", str(part)]:
            identifier = f"p_{part}"
        case ["subparts", str(subpart)]:
            identifier = f"sp_{subpart}"
        case ["subparts", str(subpart), "voices", str(voice)]:
            identifier = f"v_{voice}"
        case _:
            print("UNKNOWN URL PATH", path)

    return EntityId(identifier, href)


def tax_item_to_id(item, tax_id: str):
    id_ = item["id"]
    if item.get("name") == "na":
        id_ = "na"
    return f"{tax_id}_{id_}"


def _extract(  # noqa: C901
    obj,
    attr: str,
    type: Literal["str", "int", "entity", "taxItem"] = "str",
    taxonomy: Optional[str] = None,
) -> Any:
    if "." in attr:
        *path, attr = attr.split(".")
        for key in path:
            try:
                obj = obj[key]
            except KeyError:
                print(obj.keys())
                return None
    try:
        value = obj[attr]
    except KeyError:
        print(obj.keys())
        raise
    if type == "str":
        assert isinstance(value, str) or value is None
        return value
    if type == "int":
        assert isinstance(value, int) or value is None
        if value < 0:
            return None
        return value
    if type == "entity":
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            return [None if v.get("id", -1) < 0 else entity_to_id(v).id_ for v in value]
        if value.get("id", -1) < 0:
            return None
        return entity_to_id(value).id_
    if type == "taxItem":
        if isinstance(value, (tuple, list)):
            tax_id = f"t_{taxonomy}"
            return [tax_item_to_id(v, tax_id) for v in value if v["id"] >= 0]
        if value:
            if value["id"] < 0:
                return None
            return tax_item_to_id(value, f"t_{taxonomy}")
        return None


class PersonEntity(NamedTuple):
    ID: str
    href: str
    name: str
    birth_year: Optional[int]
    death_year: Optional[int]
    gender: str
    nationality: str


def person_to_entity(entity):
    id_, href = entity_to_id(entity)

    match entity:
        case {
            "name": str(name),
            "birth_date": int(birth),
            "death_date": int(death),
            "gender": str(gender),
            "nationality": str(nationality),
        }:
            if birth == -1:
                birth = None
            if death == -1:
                death = None
            return PersonEntity(id_, href, name, birth, death, gender, nationality)

    raise ValueError("Given Entity does not match the shape of person entities!")


class OpusEntity(NamedTuple):
    ID: str
    href: str
    opus_name: str
    original_name: str
    composer: Annotated[str, {"ref_target": "people.csv"}]
    genre: Annotated[str, {"taxonomy": "Gattung"}]
    grundton: Annotated[str, {"taxonomy": "Grundton"}]
    tonalitaet: Annotated[str, {"taxonomy": "Tonalitaet"}]
    movements: Optional[int]
    composition_year: Optional[int]
    composition_place: Optional[str]
    notes: Optional[str]
    score_link: Optional[str]
    first_printed_at: Optional[str]
    first_printed_in: Optional[str]
    first_played_at: Optional[str]
    first_played_in: Optional[str]


OPUS_FIELDS = {
    "name",
    "composer",
    "genre",
    "grundton",
    "tonalitaet",
    "original_name",
    "movements",
    "composition_year",
    "composition_place",
    "notes",
    "score_link",
    "first_printed_at",
    "first_printed_in",
    "first_played_at",
    "first_played_in",
}


def opus_to_entity(entity):
    id_, href = entity_to_id(entity)

    if entity.keys() < OPUS_FIELDS:
        raise ValueError("Given Entity does not match the shape of opus entities!")

    return OpusEntity(
        id_,
        href,
        opus_name=_extract(entity, "name"),
        original_name=_extract(entity, "original_name"),
        composer=_extract(entity, "composer", "entity"),
        genre=_extract(entity, "genre", "taxItem", "Gattung"),
        grundton=_extract(entity, "grundton", "taxItem", "Grundton"),
        tonalitaet=_extract(entity, "tonalitaet", "taxItem", "Tonalitaet"),
        movements=_extract(entity, "movements", "int"),
        composition_year=_extract(entity, "composition_year", "int"),
        composition_place=_extract(entity, "composition_place"),
        notes=_extract(entity, "notes"),
        score_link=_extract(entity, "score_link"),
        first_printed_at=_extract(entity, "first_printed_at"),
        first_printed_in=_extract(entity, "first_printed_in", "int"),
        first_played_at=_extract(entity, "first_played_at"),
        first_played_in=_extract(entity, "first_played_in", "int"),
    )


class PartEntity(NamedTuple):
    ID: str
    href: str
    part_name: str
    opus: Annotated[str, {"ref_target": "opuses.csv"}]
    movement: Optional[int]
    length: Optional[int]
    measure_start: Optional[int]
    measure_start_ref_page: Optional[int]
    measure_end: Optional[int]
    measure_end_ref_page: Optional[int]
    occurence_in_movement: Annotated[Optional[str], {"taxonomy": "AuftretenSatz"}]
    formal_functions: Annotated[List[str], {"taxonomy": "FormaleFunktion"}]
    instrument_quantity_before: Annotated[
        Optional[str], {"taxonomy": "InstrumentierungEinbettungQuantitaet"}
    ]
    instrument_quantity_after: Annotated[
        Optional[str], {"taxonomy": "InstrumentierungEinbettungQuantitaet"}
    ]
    instrument_quality_before: Annotated[
        Optional[str], {"taxonomy": "InstrumentierungEinbettungQualitaet"}
    ]
    instrument_quality_after: Annotated[
        Optional[str], {"taxonomy": "InstrumentierungEinbettungQualitaet"}
    ]
    loudness_before: Annotated[Optional[str], {"taxonomy": "Lautstaerke"}]
    loudness_after: Annotated[Optional[str], {"taxonomy": "Lautstaerke"}]
    dynamic_trend_before: Annotated[Optional[str], {"taxonomy": "LautstaerkeEinbettung"}]
    dynamic_trend_after: Annotated[Optional[str], {"taxonomy": "LautstaerkeEinbettung"}]
    tempo_before: Annotated[Optional[str], {"taxonomy": "TempoEinbettung"}]
    tempo_after: Annotated[Optional[str], {"taxonomy": "TempoEinbettung"}]
    tempo_trend_before: Annotated[Optional[str], {"taxonomy": "TempoEntwicklung"}]
    tempo_trend_after: Annotated[Optional[str], {"taxonomy": "TempoEntwicklung"}]
    ambitus_before: Annotated[Optional[str], {"taxonomy": "AmbitusEinbettung"}]
    ambitus_after: Annotated[Optional[str], {"taxonomy": "AmbitusEinbettung"}]
    ambitus_change_before: Annotated[Optional[str], {"taxonomy": "AmbitusEntwicklung"}]
    ambitus_change_after: Annotated[Optional[str], {"taxonomy": "AmbitusEntwicklung"}]
    melodic_line_before: Annotated[Optional[str], {"taxonomy": "Melodiebewegung"}]
    melodic_line_after: Annotated[Optional[str], {"taxonomy": "Melodiebewegung"}]


PART_FIELDS = {
    "opus_id",
    "name",
    "movement",
    "length",
    "measure_start",
    "measure_end",
    "occurence_in_movement",
    "formal_functions",
    "instrumentation_context",
    "dynamic_context",
    "tempo_context",
    "dramaturgic_context",
}


def part_to_entity(entity: Dict):
    id_, href = entity_to_id(entity)

    if entity.keys() < PART_FIELDS:
        raise ValueError("Given Entity does not match the shape of part entities!")

    return PartEntity(
        id_,
        href,
        part_name=_extract(entity, "name"),
        opus=f'o_{_extract(entity, "opus_id", "int")}',
        movement=_extract(entity, "movement", "int"),
        length=_extract(entity, "length", "int"),
        measure_start=_extract(entity, "measure_start.measure", "int"),
        measure_start_ref_page=_extract(entity, "measure_start.from_page", "int"),
        measure_end=_extract(entity, "measure_end.measure", "int"),
        measure_end_ref_page=_extract(entity, "measure_end.from_page", "int"),
        occurence_in_movement=_extract(
            entity, "occurence_in_movement", "taxItem", "AuftretenSatz"
        ),
        formal_functions=_extract(
            entity, "formal_functions", "taxItem", "FormaleFunktion"
        ),
        instrument_quantity_before=_extract(
            entity,
            "instrumentation_context.instrumentation_quantity_before",
            "taxItem",
            "InstrumentierungEinbettungQuantitaet",
        ),
        instrument_quantity_after=_extract(
            entity,
            "instrumentation_context.instrumentation_quantity_after",
            "taxItem",
            "InstrumentierungEinbettungQuantitaet",
        ),
        instrument_quality_before=_extract(
            entity,
            "instrumentation_context.instrumentation_quality_before",
            "taxItem",
            "InstrumentierungEinbettungQualitaet",
        ),
        instrument_quality_after=_extract(
            entity,
            "instrumentation_context.instrumentation_quality_after",
            "taxItem",
            "InstrumentierungEinbettungQualitaet",
        ),
        loudness_before=_extract(
            entity, "dynamic_context.loudness_before", "taxItem", "Lautstaerke"
        ),
        loudness_after=_extract(
            entity, "dynamic_context.loudness_after", "taxItem", "Lautstaerke"
        ),
        dynamic_trend_before=_extract(
            entity,
            "dynamic_context.dynamic_trend_before",
            "taxItem",
            "LautstaerkeEinbettung",
        ),
        dynamic_trend_after=_extract(
            entity,
            "dynamic_context.dynamic_trend_after",
            "taxItem",
            "LautstaerkeEinbettung",
        ),
        tempo_before=_extract(
            entity, "tempo_context.tempo_context_before", "taxItem", "TempoEinbettung"
        ),
        tempo_after=_extract(
            entity, "tempo_context.tempo_context_after", "taxItem", "TempoEinbettung"
        ),
        tempo_trend_before=_extract(
            entity, "tempo_context.tempo_trend_before", "taxItem", "TempoEntwicklung"
        ),
        tempo_trend_after=_extract(
            entity, "tempo_context.tempo_trend_after", "taxItem", "TempoEntwicklung"
        ),
        ambitus_before=_extract(
            entity,
            "dramaturgic_context.ambitus_context_before",
            "taxItem",
            "AmbitusEinbettung",
        ),
        ambitus_after=_extract(
            entity,
            "dramaturgic_context.ambitus_context_after",
            "taxItem",
            "AmbitusEinbettung",
        ),
        ambitus_change_before=_extract(
            entity,
            "dramaturgic_context.ambitus_change_before",
            "taxItem",
            "AmbitusEntwicklung",
        ),
        ambitus_change_after=_extract(
            entity,
            "dramaturgic_context.ambitus_change_after",
            "taxItem",
            "AmbitusEntwicklung",
        ),
        melodic_line_before=_extract(
            entity,
            "dramaturgic_context.melodic_line_before",
            "taxItem",
            "Melodiebewegung",
        ),
        melodic_line_after=_extract(
            entity, "dramaturgic_context.melodic_line_after", "taxItem", "Melodiebewegung"
        ),
    )


class SubpartEntity(NamedTuple):
    ID: str
    href: str
    subpart_label: str
    part: Annotated[str, {"ref_target": "parts.csv"}]
    opus: Annotated[str, {"ref_target": "opuses.csv"}]
    is_tutti: bool
    occurence_in_part: Annotated[Optional[str], {"taxonomy": "AuftretenWerkausschnitt"}]
    share_of_part: Annotated[Optional[str], {"taxonomy": "Anteil"}]
    instrumentation: Annotated[List[str], {"taxonomy": "Instrument"}]
    tempo_markings: Annotated[List[str], {"taxonomy": "Tempo"}]
    tempo_changes: Annotated[List[str], {"taxonomy": "TempoEntwicklung"}]
    dynamic_markings: Annotated[List[Optional[str]], {"taxonomy": "Lautstaerke"}]
    dynamic_markings_extra: Annotated[
        List[Optional[str]], {"taxonomy": "LautstaerkeZusatz"}
    ]
    dynamic_changes: Annotated[List[str], {"taxonomy": "LautstaerkeEntwicklung"}]
    degree_of_dissonance: Annotated[Optional[str], {"taxonomy": "Dissonanzgrad"}]
    dissonances: Annotated[List[str], {"taxonomy": "Dissonanzen"}]
    chords: Annotated[List[str], {"taxonomy": "Akkord"}]
    harmonic_complexity: Annotated[Optional[str], {"taxonomy": "HarmonischeKomplexitaet"}]
    harmonic_density: Annotated[Optional[str], {"taxonomy": "HarmonischeDichte"}]
    harmonic_phenomenons: Annotated[List[str], {"taxonomy": "HarmonischePhaenomene"}]
    harmonic_changes: Annotated[List[str], {"taxonomy": "HarmonischeEntwicklung"}]
    harmonische_function: Annotated[
        List[str], {"taxonomy": "HarmonischeFunktionVerwandschaft"}
    ]
    harmonic_analysis: Optional[str]
    hc__tonalitaet: Annotated[List[Optional[str]], {"taxonomy": "Tonalitaet"}]
    hc__harmonic_function: Annotated[
        List[Optional[str]], {"taxonomy": "HarmonischeFunktion"}
    ]
    hc__grundton: Annotated[List[Optional[str]], {"taxonomy": "Grundton"}]
    hc__harmonic_step: Annotated[List[Optional[str]], {"taxonomy": "HarmonischeStufe"}]


SUBPART_FIELDS = {
    "part_id",
    "label",
    "is_tutti",
    "occurence_in_part",
    "share_of_part",
    "instrumentation",
    "tempo",
    "dynamic",
    "harmonics",
}


def subpart_to_entity(entity, part_id_to_opus_id: Dict[str, str]):
    id_, href = entity_to_id(entity)

    if entity.keys() < SUBPART_FIELDS:
        raise ValueError("Given Entity does not match the shape of subpart entities!")

    part_id = f'p_{_extract(entity, "part_id", "int")}'
    dynamic_markings = entity.get("dynamic", {}).get("dynamic_markings", [])
    harmonic_centers = entity.get("harmonics", {}).get("harmonic_centers", [])

    return SubpartEntity(
        id_,
        href,
        subpart_label=_extract(entity, "label"),
        part=part_id,
        opus=part_id_to_opus_id[part_id],
        is_tutti=bool(entity.get("is_tutti")),
        occurence_in_part=_extract(
            entity, "occurence_in_part", "taxItem", "AuftretenWerkausschnitt"
        ),
        share_of_part=_extract(entity, "share_of_part", "taxItem", "Anteil"),
        instrumentation=_extract(entity, "instrumentation", "taxItem", "Instrument"),
        tempo_markings=_extract(entity, "tempo.tempo_markings", "taxItem", "Tempo"),
        tempo_changes=_extract(
            entity, "tempo.tempo_changes", "taxItem", "TempoEntwicklung"
        ),
        dynamic_markings=[
            _extract(e, "lautstaerke", "taxItem", "Lautstaerke") for e in dynamic_markings
        ],
        dynamic_markings_extra=[
            _extract(e, "lautstaerke_zusatz", "taxItem", "LautstaerkeZusatz")
            for e in dynamic_markings
        ],
        dynamic_changes=_extract(
            entity, "dynamic.dynamic_changes", "taxItem", "LautstaerkeEntwicklung"
        ),
        degree_of_dissonance=_extract(
            entity, "harmonics.degree_of_dissonance", "taxItem", "Dissonanzgrad"
        ),
        dissonances=_extract(entity, "harmonics.dissonances", "taxItem", "Dissonanzen"),
        chords=_extract(entity, "harmonics.chords", "taxItem", "Akkord"),
        harmonic_complexity=_extract(
            entity, "harmonics.harmonic_complexity", "taxItem", "HarmonischeKomplexitaet"
        ),
        harmonic_density=_extract(
            entity, "harmonics.harmonic_density", "taxItem", "HarmonischeDichte"
        ),
        harmonic_phenomenons=_extract(
            entity, "harmonics.harmonic_phenomenons", "taxItem", "HarmonischePhaenomene"
        ),
        harmonic_changes=_extract(
            entity, "harmonics.harmonic_changes", "taxItem", "HarmonischeEntwicklung"
        ),
        harmonische_function=_extract(
            entity,
            "harmonics.harmonische_funktion",
            "taxItem",
            "HarmonischeFunktionVerwandschaft",
        ),
        harmonic_analysis=_extract(entity, "harmonics.harmonic_analyse"),
        hc__tonalitaet=[
            _extract(e, "tonalitaet", "taxItem", "Tonalitaet") for e in harmonic_centers
        ],
        hc__harmonic_function=[
            _extract(e, "harmonische_funktion", "taxItem", "HarmonischeFunktion")
            for e in harmonic_centers
        ],
        hc__grundton=[
            _extract(e, "grundton", "taxItem", "Grundton") for e in harmonic_centers
        ],
        hc__harmonic_step=[
            _extract(e, "harmonische_stufe", "taxItem", "HarmonischeStufe")
            for e in harmonic_centers
        ],
    )


class VoiceEntity(NamedTuple):
    ID: str
    href: str
    name: str
    subpart: Annotated[str, {"ref_target": "subparts.csv"}]
    part: Annotated[str, {"ref_target": "parts.csv"}]
    opus: Annotated[str, {"ref_target": "opuses.csv"}]
    measure_start: Optional[int]
    measure_start_ref_page: Optional[int]
    measure_end: Optional[int]
    measure_end_ref_page: Optional[int]
    instrumentation: Annotated[List[str], {"taxonomy": "Instrument"}]
    has_melody: bool
    musicial_function: Annotated[List[str], {"taxonomy": "MusikalischeFunktion"}]
    share_of_subpart: Annotated[Optional[str], {"taxonomy": "Anteil"}]
    occurence_in_subpart: Annotated[
        Optional[str], {"taxonomy": "AuftretenWerkausschnitt"}
    ]
    dominant_note_values: Annotated[List[str], {"taxonomy": "Notenwert"}]
    musicial_figures: Annotated[List[str], {"taxonomy": "MusikalischeWendung"}]
    ornaments: Annotated[List[str], {"taxonomy": "Verzierung"}]
    melody_form: Annotated[Optional[str], {"taxonomy": "Melodieform"}]
    intervallik: Annotated[List[str], {"taxonomy": "Intervallik"}]
    highest_pitch: Annotated[Optional[str], {"taxonomy": "Grundton"}]
    highest_octave: Annotated[Optional[str], {"taxonomy": "Oktave"}]
    lowest_pitch: Annotated[Optional[str], {"taxonomy": "Grundton"}]
    lowest_octave: Annotated[Optional[str], {"taxonomy": "Oktave"}]
    satzart_allgemein: Annotated[List[str], {"taxonomy": "SatzartAllgemein"}]
    satzart_speziell: Annotated[List[str], {"taxonomy": "SatzartSpeziell"}]
    time_signatures: Annotated[List[str], {"taxonomy": "Taktart"}]
    rhythmic_phenomenons: Annotated[List[str], {"taxonomy": "RhythmischesPhaenomen"}]
    rhythm_types: Annotated[List[str], {"taxonomy": "Rhythmustyp"}]
    is_polymetric: bool
    nr_repetitions_1_2: int
    nr_repetitions_3_4: int
    nr_repetitions_5_6: int
    nr_repetitions_7_10: int
    composition_techniques: Annotated[List[str], {"taxonomy": "Verarbeitungstechnik"}]
    sequence_beats: List[int]
    sequence_flow: Annotated[List[Optional[str]], {"taxonomy": "BewegungImTonraum"}]
    sequence_is_exact_repetition: List[bool]
    sequence_is_tonal_corrected: List[bool]
    sequence_starting_intervall: Annotated[List[Optional[str]], {"taxonomy": "Intervall"}]
    mood_markings: Annotated[List[str], {"taxonomy": "Ausdruck"}]
    technic_markings: Annotated[List[str], {"taxonomy": "Spielanweisung"}]
    articulation_markings: Annotated[List[str], {"taxonomy": "Artikulation"}]
    cited_opus: Annotated[List[str], {"ref_target": "opuses.csv"}]
    opus_citation_kind: Annotated[List[Optional[str]], {"taxonomy": "Zitat"}]
    cited_genre: Annotated[List[str], {"taxonomy": "Gattung"}]
    cited_instrument: Annotated[List[str], {"taxonomy": "Instrument"}]
    cited_composer: Annotated[List[str], {"ref_target": "people.csv"}]
    cited_program_theme: Annotated[List[str], {"taxonomy": "Programmgegenstand"}]
    cited_sound: Annotated[List[str], {"taxonomy": "Tonmalerei"}]
    cited_epoch: Annotated[List[str], {"taxonomy": "Epoche"}]
    related_voices: Annotated[List[str], {"ref_target": "voices.csv"}]
    voice_relation_kind: Annotated[
        List[Optional[str]], {"taxonomy": "VoiceToVoiceRelation"}
    ]


VOICE_FIELDS = {
    "subpart_id",
    "name",
    "instrumentation",
    "ambitus",
    "has_melody",
    "musicial_function",
    "share",
    "occurence_in_part",
    "satz",
    "rhythm",
    "dominant_note_values",
    "composition",
    "musicial_figures",
    "rendition",
    "ornaments",
    "melody_form",
    "intervallik",
    "citations",
    "related_voices",
    "measure_start",
    "measure_end",
}


def voice_to_entity(
    entity, subpart_id_to_part_id: Dict[str, str], part_id_to_opus_id: Dict[str, str]
):
    id_, href = entity_to_id(entity)

    if entity.keys() < VOICE_FIELDS:
        raise ValueError("Given Entity does not match the shape of voice entities!")

    subpart_id = f'sp_{_extract(entity, "subpart_id", "int")}'
    part_id = subpart_id_to_part_id[subpart_id]

    sequences = entity.get("composition", {}).get("sequences", [])
    opus_citations = [
        c
        for c in entity.get("citations", {}).get("opus_citations", [])
        if c.get("opus", {}).get("id", -1) >= 0
    ]

    related_voices = [
        v
        for v in entity.get("related_voices", [])
        if v.get("related_voice", {}).get("id", -1) >= 0
    ]

    return VoiceEntity(
        id_,
        href,
        name=_extract(entity, "name"),
        subpart=subpart_id,
        part=part_id,
        opus=part_id_to_opus_id[part_id],
        measure_start=_extract(entity, "measure_start.measure", "int"),
        measure_start_ref_page=_extract(entity, "measure_start.from_page", "int"),
        measure_end=_extract(entity, "measure_end.measure", "int"),
        measure_end_ref_page=_extract(entity, "measure_end.from_page", "int"),
        instrumentation=_extract(entity, "instrumentation", "taxItem", "Instrument"),
        has_melody=bool(entity.get("has_melody", False)),
        musicial_function=_extract(
            entity, "musicial_function", "taxItem", "MusikalischeFunktion"
        ),
        share_of_subpart=_extract(entity, "share", "taxItem", "Anteil"),
        occurence_in_subpart=_extract(
            entity, "occurence_in_part", "taxItem", "AuftretenWerkausschnitt"
        ),
        dominant_note_values=_extract(
            entity, "dominant_note_values", "taxItem", "Notenwert"
        ),
        musicial_figures=_extract(
            entity, "musicial_figures", "taxItem", "MusikalischeWendung"
        ),
        ornaments=_extract(entity, "ornaments", "taxItem", "Verzierung"),
        melody_form=_extract(entity, "melody_form", "taxItem", "Melodieform"),
        intervallik=_extract(entity, "intervallik", "taxItem", "Intervallik"),
        highest_pitch=_extract(entity, "ambitus.highest_pitch", "taxItem", "Grundton"),
        highest_octave=_extract(entity, "ambitus.highest_octave", "taxItem", "Oktave"),
        lowest_pitch=_extract(entity, "ambitus.lowest_pitch", "taxItem", "Grundton"),
        lowest_octave=_extract(entity, "ambitus.lowest_octave", "taxItem", "Oktave"),
        satzart_allgemein=_extract(
            entity, "satz.satzart_allgemein", "taxItem", "SatzartAllgemein"
        ),
        satzart_speziell=_extract(
            entity, "satz.satzart_speziell", "taxItem", "SatzartSpeziell"
        ),
        time_signatures=_extract(entity, "rhythm.measure_times", "taxItem", "Taktart"),
        rhythmic_phenomenons=_extract(
            entity, "rhythm.rhythmic_phenomenons", "taxItem", "RhythmischesPhaenomen"
        ),
        rhythm_types=_extract(entity, "rhythm.rhythm_types", "taxItem", "Rhythmustyp"),
        is_polymetric=bool(entity.get("rhythm", {}).get("polymetric", False)),
        nr_repetitions_1_2=_extract(entity, "composition.nr_repetitions_1_2", "int"),
        nr_repetitions_3_4=_extract(entity, "composition.nr_repetitions_3_4", "int"),
        nr_repetitions_5_6=_extract(entity, "composition.nr_repetitions_5_6", "int"),
        nr_repetitions_7_10=_extract(entity, "composition.nr_repetitions_7_10", "int"),
        composition_techniques=_extract(
            entity,
            "composition.composition_techniques",
            "taxItem",
            "Verarbeitungstechnik",
        ),
        sequence_beats=[_extract(s, "beats", "int") for s in sequences],
        sequence_flow=[
            _extract(s, "flow", "taxItem", "BewegungImTonraum") for s in sequences
        ],
        sequence_is_exact_repetition=[
            bool(s.get("exact_repetition", False)) for s in sequences
        ],
        sequence_is_tonal_corrected=[
            bool(s.get("tonal_corrected", False)) for s in sequences
        ],
        sequence_starting_intervall=[
            _extract(s, "flow", "taxItem", "Intervall") for s in sequences
        ],
        mood_markings=_extract(entity, "rendition.mood_markings", "taxItem", "Ausdruck"),
        technic_markings=_extract(
            entity, "rendition.technic_markings", "taxItem", "Spielanweisung"
        ),
        articulation_markings=_extract(
            entity, "rendition.articulation_markings", "taxItem", "Artikulation"
        ),
        cited_opus=[entity_to_id(c["opus"]).id_ for c in opus_citations],
        opus_citation_kind=[
            _extract(c, "citation_type", "taxItem", "Zitat") for c in opus_citations
        ],
        cited_genre=_extract(entity, "citations.gattung_citations", "taxItem", "Gattung"),
        cited_instrument=_extract(
            entity, "citations.instrument_citations", "taxItem", "Instrument"
        ),
        cited_composer=_extract(entity, "citations.composer_citations", "entity"),
        cited_program_theme=_extract(
            entity, "citations.program_citations", "taxItem", "Programmgegenstand"
        ),
        cited_sound=_extract(
            entity, "citations.tonmalerei_citations", "taxItem", "Tonmalerei"
        ),
        cited_epoch=_extract(entity, "citations.epoch_citations", "taxItem", "Epoche"),
        related_voices=[entity_to_id(r["related_voice"]).id_ for r in related_voices],
        voice_relation_kind=[
            _extract(r, "type_of_relationship", "taxItem", "VoiceToVoiceRelation")
            for r in related_voices
        ],
    )


class VoiceRelation(NamedTuple):
    source: str
    target: str
    voice_relation_kind: Annotated[Optional[str], {"taxonomy": "VoiceToVoiceRelation"}]


def voices_to_voice_relation_graph(voices: Sequence[VoiceEntity]):
    entities = []
    relations = []

    for voice in voices:
        entities.append(voice.ID)
        for target, kind in zip(voice.related_voices, voice.voice_relation_kind):
            if target is None:
                continue
            relations.append(VoiceRelation(voice.ID, target, kind))
    return {
        "GRAPH_ID": "voice-relations",
        "type": "directed",
        "ref-target": "voices.csv",
        "entities": entities,
        "relations": relations,
    }


class OpusCitationRelation(NamedTuple):
    source: str
    target: str
    via: Annotated[str, {"ref_target": "voices.csv"}]
    opus_citation_kind: Annotated[Optional[str], {"taxonomy": "Zitat"}]


def voices_to_opus_citation_graph(voices: Sequence[VoiceEntity]):
    entities = set()
    relations = []

    for voice in voices:
        entities.add(voice.opus)
        for target, kind in zip(voice.cited_opus, voice.opus_citation_kind):
            if target is None:
                continue
            entities.add(target)
            relations.append(OpusCitationRelation(voice.opus, target, voice.ID, kind))
        for target in voice.cited_composer:
            if target is None:
                continue
            entities.add(target)
            relations.append(OpusCitationRelation(voice.opus, target, voice.ID, None))
    return {
        "GRAPH_ID": "opus-citations",
        "type": "directed",
        "ref-target": ["opuses.csv", "pople.csv"],
        "entities": sorted(entities),
        "relations": relations,
    }


class TaxonomyEntity(NamedTuple):
    GRAPH_ID: str
    type: str
    ref_target: str
    entities: List[Union[str, dict]]
    relations: List[Dict[str, str]]

    def to_dict(self):
        dictionary = self._asdict()
        dictionary["ref-target"] = dictionary.pop("ref_target")

        return dictionary


def _parse_tree_node(
    item, tax_id: str, na_item: Optional[Dict] = None
) -> Tuple[List[dict], List[Dict[str, str]]]:
    id_ = tax_item_to_id(item, tax_id)
    name = item["name"]
    entities = [
        {"ID": id_, "tax_item_name": name, "description": item.get("description", "")}
    ]
    relations = []

    if na_item:
        na_id = tax_item_to_id(na_item, tax_id)
        entities.append(
            {
                "ID": na_id,
                "tax_item_name": "na",
                "description": na_item.get("description", ""),
            }
        )
        # assume first node as root
        relations.append({"source": id_, "target": na_id})

    for child in item["children"]:
        relations.append({"source": id_, "target": tax_item_to_id(child, tax_id)})

        new_entities, new_relations = _parse_tree_node(child, tax_id)
        entities.extend(new_entities)
        relations.extend(new_relations)

    return entities, relations


def _parse_list_to_tree(
    items: List, tax_id: str, na_item: Optional[Dict] = None
) -> Tuple[List[dict], List[Dict[str, str]]]:
    root_id = f"{tax_id}_root"
    entities = [{"ID": root_id, "tax_item_name": "root", "description": ""}]
    relations = []

    if na_item:
        na_id = tax_item_to_id(na_item, tax_id)
        entities.append(
            {
                "ID": na_id,
                "tax_item_name": "na",
                "description": na_item.get("description", ""),
            }
        )
        relations.append({"source": root_id, "target": na_id})

    for item in items:
        id_ = tax_item_to_id(item, tax_id)
        name = item["name"]
        entities.append(
            {"ID": id_, "tax_item_name": name, "description": item.get("description", "")}
        )
        relations.append({"source": root_id, "target": id_})

    return entities, relations


def taxonomy_to_entity(
    entity: Dict,
) -> TaxonomyEntity:
    graph_id = entity_to_id(entity).id_
    tax_type = "tree"
    ref_target = "taxonomies.zip"

    tax_id = entity_to_id(entity).id_

    na_item = entity.get("na_item", None)

    if entity["taxonomy_type"] == "tree":
        entities, relations = _parse_tree_node(entity["items"], tax_id, na_item=na_item)
    elif entity["taxonomy_type"] == "list":
        # gets handled like a tree with depth 1
        entities, relations = _parse_list_to_tree(
            entity["items"], tax_id, na_item=na_item
        )
    else:
        raise ValueError(f"Unknown taxonomy type {entity['taxonomy_type']}")

    return TaxonomyEntity(graph_id, tax_type, ref_target, list(entities), relations)


def _unwrap_optional(type_hint: type) -> type:
    if get_origin(type_hint) == Union:
        return get_args(type_hint)[0]

    return type_hint


def _entity_class_to_attribute_metadata(entity_class: type) -> List[AttributeMetadata]:
    type_hints = get_type_hints(entity_class, include_extras=True)
    metadata = []

    for attribute, type_hint in type_hints.items():
        if attribute in ("ID", "href"):
            continue

        type_name = None
        ref_target = None
        taxonomy = None
        multiple = False

        if get_origin(type_hint) == Annotated:
            type_meta = type_hint.__metadata__[0]
            type_hint = get_args(type_hint)[0]

            if "ref_target" in type_meta:
                ref_target = type_meta["ref_target"]
                type_name = "ref"

            if "taxonomy" in type_meta:
                taxonomy = type_meta["taxonomy"]
                type_name = "ref"

        type_hint = _unwrap_optional(type_hint)

        if get_origin(type_hint) == List or get_origin(type_hint) == list:
            multiple = True
            type_hint = get_args(type_hint)[0]

        if type_name != "ref":
            if type_hint == str:
                type_name = "string"

            if type_hint == int:
                type_name = "integer"

            if type_hint == float:
                type_name = "number"

            if type_hint == bool:
                type_name = "boolean"

        if type_name is None:
            raise ValueError("Cannot use None as type name!")

        metadata.append(
            AttributeMetadata(
                attribute,
                attribute,
                "",
                type_name,
                multiple,
                False,
                ";",
                f"taxonomies.zip:t_{taxonomy}.json" if taxonomy else ref_target,
                None,
                {},
            )
        )

    return metadata


def get_attribute_metadata() -> Dict[str, AttributeMetadata]:
    metadata = []

    metadata.extend(_entity_class_to_attribute_metadata(PersonEntity))
    metadata.extend(_entity_class_to_attribute_metadata(OpusEntity))
    metadata.extend(_entity_class_to_attribute_metadata(PartEntity))
    metadata.extend(_entity_class_to_attribute_metadata(SubpartEntity))
    metadata.extend(_entity_class_to_attribute_metadata(VoiceEntity))

    deduplicated_metadata = {}

    for meta in metadata:
        deduplicated_metadata[meta.ID] = meta

    return deduplicated_metadata
