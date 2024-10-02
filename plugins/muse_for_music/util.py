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

from typing import Any, List, Literal, NamedTuple, Optional, Dict
from urllib.parse import urlparse


class EntityId(NamedTuple):
    id_: str
    href: str


def entity_to_id(entity) -> EntityId:
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


def _extract(
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
        return entity_to_id(value).id_
    if type == "taxItem":
        # TODO: handle not applicable values???
        if isinstance(value, (tuple, list)):
            return [f't_{taxonomy}_{v["id"]}' for v in value if v["id"] >= 0]
        if value:
            if value["id"] < 0:
                return None
            return f't_{taxonomy}_{value["id"]}'
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
    name: str
    original_name: str
    composer: str
    genre: str
    grundton: str
    tonalitaet: str
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

    name = _extract(entity, "name")
    original_name = _extract(entity, "original_name")
    composer = _extract(entity, "composer", "entity")
    genre = _extract(entity, "genre", "taxItem", "Gattung")
    grundton = _extract(entity, "grundton", "taxItem", "Grundton")
    tonalitaet = _extract(entity, "tonalitaet", "taxItem", "Tonalitaet")
    movements = _extract(entity, "movements", "int")
    composition_year = _extract(entity, "composition_year", "int")
    composition_place = _extract(entity, "composition_place")
    notes = _extract(entity, "notes")
    score_link = _extract(entity, "score_link")
    first_printed_at = _extract(entity, "first_printed_at")
    first_printed_in = _extract(entity, "first_printed_in", "int")
    first_played_at = _extract(entity, "first_played_at")
    first_played_in = _extract(entity, "first_played_in", "int")

    return OpusEntity(
        id_,
        href,
        name,
        original_name,
        composer,
        genre,
        grundton,
        tonalitaet,
        movements,
        composition_year,
        composition_place,
        notes,
        score_link,
        first_printed_at,
        first_printed_in,
        first_played_at,
        first_played_in,
    )


class PartEntity(NamedTuple):
    ID: str
    href: str
    name: str
    opus: str
    movement: Optional[int]
    length: Optional[int]
    measure_start: Optional[int]
    measure_start_ref_page: Optional[int]
    measure_end: Optional[int]
    measure_end_ref_page: Optional[int]
    occurence_in_movement: Optional[str]
    formal_functions: List[str]
    instrument_quantity_before: Optional[str]
    instrument_quantity_after: Optional[str]
    instrument_quality_before: Optional[str]
    instrument_quality_after: Optional[str]
    loudness_before: Optional[str]
    loudness_after: Optional[str]
    dynamic_trend_before: Optional[str]
    dynamic_trend_after: Optional[str]
    tempo_before: Optional[str]
    tempo_after: Optional[str]
    tempo_trend_before: Optional[str]
    tempo_trend_after: Optional[str]
    ambitus_before: Optional[str]
    ambitus_after: Optional[str]
    ambitus_change_before: Optional[str]
    ambitus_change_after: Optional[str]
    melodic_line_before: Optional[str]
    melodic_line_after: Optional[str]


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


def part_to_entity(entity):
    id_, href = entity_to_id(entity)

    if entity.keys() < PART_FIELDS:
        raise ValueError("Given Entity does not match the shape of part entities!")

    name = _extract(entity, "name")
    opus = f'o_{_extract(entity, "opus_id", "int")}'
    movement = _extract(entity, "movement", "int")
    length = _extract(entity, "length", "int")
    measure_start = _extract(entity, "measure_start.measure", "int")
    measure_start_page = _extract(entity, "measure_start.from_page", "int")
    measure_end = _extract(entity, "measure_end.measure", "int")
    measure_end_page = _extract(entity, "measure_end.from_page", "int")
    occurence_in_movement = _extract(
        entity, "occurence_in_movement", "taxItem", "AuftretenSatz"
    )
    formal_functions = _extract(entity, "formal_functions", "taxItem", "FormaleFunktion")

    instr_quant_before = _extract(
        entity,
        "instrumentation_context.instrumentation_quantity_before",
        "taxItem",
        "InstrumentierungEinbettungQuantitaet",
    )
    instr_quant_after = _extract(
        entity,
        "instrumentation_context.instrumentation_quantity_after",
        "taxItem",
        "InstrumentierungEinbettungQuantitaet",
    )
    instr_qual_before = _extract(
        entity,
        "instrumentation_context.instrumentation_quality_before",
        "taxItem",
        "InstrumentierungEinbettungQualitaet",
    )
    instr_qual_after = _extract(
        entity,
        "instrumentation_context.instrumentation_quality_after",
        "taxItem",
        "InstrumentierungEinbettungQualitaet",
    )

    loudness_before = _extract(
        entity, "dynamic_context.loudness_before", "taxItem", "Lautstaerke"
    )
    loudness_after = _extract(
        entity, "dynamic_context.loudness_after", "taxItem", "Lautstaerke"
    )
    dynamic_trend_before = _extract(
        entity, "dynamic_context.dynamic_trend_before", "taxItem", "LautstaerkeEinbettung"
    )
    dynamic_trend_after = _extract(
        entity, "dynamic_context.dynamic_trend_after", "taxItem", "LautstaerkeEinbettung"
    )

    tempo_before = _extract(
        entity, "tempo_context.tempo_context_before", "taxItem", "TempoEinbettung"
    )
    tempo_after = _extract(
        entity, "tempo_context.tempo_context_after", "taxItem", "TempoEinbettung"
    )
    tempo_trend_before = _extract(
        entity, "tempo_context.tempo_trend_before", "taxItem", "TempoEntwicklung"
    )
    tempo_trend_after = _extract(
        entity, "tempo_context.tempo_trend_after", "taxItem", "TempoEntwicklung"
    )

    ambitus_before = _extract(
        entity,
        "dramaturgic_context.ambitus_context_before",
        "taxItem",
        "AmbitusEinbettung",
    )
    ambitus_after = _extract(
        entity,
        "dramaturgic_context.ambitus_context_after",
        "taxItem",
        "AmbitusEinbettung",
    )
    ambitus_change_before = _extract(
        entity,
        "dramaturgic_context.ambitus_change_before",
        "taxItem",
        "AmbitusEntwicklung",
    )
    ambitus_change_after = _extract(
        entity,
        "dramaturgic_context.ambitus_change_after",
        "taxItem",
        "AmbitusEntwicklung",
    )
    melodic_line_before = _extract(
        entity, "dramaturgic_context.melodic_line_before", "taxItem", "Melodiebewegung"
    )
    melodic_line_after = _extract(
        entity, "dramaturgic_context.melodic_line_after", "taxItem", "Melodiebewegung"
    )

    return PartEntity(
        id_,
        href,
        name,
        opus,
        movement,
        length,
        measure_start,
        measure_start_page,
        measure_end,
        measure_end_page,
        occurence_in_movement,
        formal_functions,
        instr_quant_before,
        instr_quant_after,
        instr_qual_before,
        instr_qual_after,
        loudness_before,
        loudness_after,
        dynamic_trend_before,
        dynamic_trend_after,
        tempo_before,
        tempo_after,
        tempo_trend_before,
        tempo_trend_after,
        ambitus_before,
        ambitus_after,
        ambitus_change_before,
        ambitus_change_after,
        melodic_line_before,
        melodic_line_after,
    )


class TaxonomyEntity(NamedTuple):
    GRAPH_ID: str
    type: str
    ref_target: str  # TODO: serialize to "ref-target"
    entities: List[str]
    relations: List[Dict[str, str]]


def _get_entities_and_relations_from_item(item):
    name = item["name"]
    entities = {name}
    relations = []

    for child in item["children"]:
        relations.append({"source": name, "target": child["name"]})

        new_entities, new_relations = _get_entities_and_relations_from_item(child)
        entities.update(new_entities)
        relations.extend(new_relations)

    return entities, relations


def taxonomy_to_entity(entity):
    if entity["taxonomy_type"] == "tree":
        graph_id = entity_to_id(entity).id_
        tax_type = "tree"
        ref_target = "entities.json"  # TODO: use correct target
        entities, relations = _get_entities_and_relations_from_item(entity["items"])

        return TaxonomyEntity(graph_id, tax_type, ref_target, list(entities), relations)
    elif entity["taxonomy_type"] == "list":
        raise NotImplemented  # TODO
