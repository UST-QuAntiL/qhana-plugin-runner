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
    Tuple,
    Union,
    get_type_hints,
    get_origin,
    get_args,
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
        _extract(entity, "name"),
        _extract(entity, "original_name"),
        _extract(entity, "composer", "entity"),
        _extract(entity, "genre", "taxItem", "Gattung"),
        _extract(entity, "grundton", "taxItem", "Grundton"),
        _extract(entity, "tonalitaet", "taxItem", "Tonalitaet"),
        _extract(entity, "movements", "int"),
        _extract(entity, "composition_year", "int"),
        _extract(entity, "composition_place"),
        _extract(entity, "notes"),
        _extract(entity, "score_link"),
        _extract(entity, "first_printed_at"),
        _extract(entity, "first_printed_in", "int"),
        _extract(entity, "first_played_at"),
        _extract(entity, "first_played_in", "int"),
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
        _extract(entity, "name"),
        f'o_{_extract(entity, "opus_id", "int")}',
        _extract(entity, "movement", "int"),
        _extract(entity, "length", "int"),
        _extract(entity, "measure_start.measure", "int"),
        _extract(entity, "measure_start.from_page", "int"),
        _extract(entity, "measure_end.measure", "int"),
        _extract(entity, "measure_end.from_page", "int"),
        _extract(entity, "occurence_in_movement", "taxItem", "AuftretenSatz"),
        _extract(entity, "formal_functions", "taxItem", "FormaleFunktion"),
        _extract(
            entity,
            "instrumentation_context.instrumentation_quantity_before",
            "taxItem",
            "InstrumentierungEinbettungQuantitaet",
        ),
        _extract(
            entity,
            "instrumentation_context.instrumentation_quantity_after",
            "taxItem",
            "InstrumentierungEinbettungQuantitaet",
        ),
        _extract(
            entity,
            "instrumentation_context.instrumentation_quality_before",
            "taxItem",
            "InstrumentierungEinbettungQualitaet",
        ),
        _extract(
            entity,
            "instrumentation_context.instrumentation_quality_after",
            "taxItem",
            "InstrumentierungEinbettungQualitaet",
        ),
        _extract(entity, "dynamic_context.loudness_before", "taxItem", "Lautstaerke"),
        _extract(entity, "dynamic_context.loudness_after", "taxItem", "Lautstaerke"),
        _extract(
            entity,
            "dynamic_context.dynamic_trend_before",
            "taxItem",
            "LautstaerkeEinbettung",
        ),
        _extract(
            entity,
            "dynamic_context.dynamic_trend_after",
            "taxItem",
            "LautstaerkeEinbettung",
        ),
        _extract(
            entity, "tempo_context.tempo_context_before", "taxItem", "TempoEinbettung"
        ),
        _extract(
            entity, "tempo_context.tempo_context_after", "taxItem", "TempoEinbettung"
        ),
        _extract(
            entity, "tempo_context.tempo_trend_before", "taxItem", "TempoEntwicklung"
        ),
        _extract(
            entity, "tempo_context.tempo_trend_after", "taxItem", "TempoEntwicklung"
        ),
        _extract(
            entity,
            "dramaturgic_context.ambitus_context_before",
            "taxItem",
            "AmbitusEinbettung",
        ),
        _extract(
            entity,
            "dramaturgic_context.ambitus_context_after",
            "taxItem",
            "AmbitusEinbettung",
        ),
        _extract(
            entity,
            "dramaturgic_context.ambitus_change_before",
            "taxItem",
            "AmbitusEntwicklung",
        ),
        _extract(
            entity,
            "dramaturgic_context.ambitus_change_after",
            "taxItem",
            "AmbitusEntwicklung",
        ),
        _extract(
            entity,
            "dramaturgic_context.melodic_line_before",
            "taxItem",
            "Melodiebewegung",
        ),
        _extract(
            entity, "dramaturgic_context.melodic_line_after", "taxItem", "Melodiebewegung"
        ),
    )


class SubpartEntity(NamedTuple):
    ID: str
    href: str
    subpart_label: str
    part: str
    opus: str
    is_tutti: bool


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

    return SubpartEntity(
        id_,
        href,
        _extract(entity, "label"),
        part_id,
        part_id_to_opus_id[part_id],
        entity["is_tutti"],
    )


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


def _parse_tree_node(item, tax_id: str) -> Tuple[List[dict], List[Dict[str, str]]]:
    # TODO: add "na" entity
    id_ = tax_item_to_id(item, tax_id)
    name = item["name"]
    entities = [
        {"ID": id_, "tax_item_name": name, "description": item.get("description", "")}
    ]
    relations = []

    for child in item["children"]:
        relations.append({"source": id_, "target": tax_item_to_id(child, tax_id)})

        new_entities, new_relations = _parse_tree_node(child, tax_id)
        entities.extend(new_entities)
        relations.extend(new_relations)

    return entities, relations


def _parse_list_to_tree(
    items: List, tax_id: str
) -> Tuple[List[dict], List[Dict[str, str]]]:
    # TODO: add "na" entity
    root_id = f"{tax_id}_root"
    entities = [{"ID": root_id, "tax_item_name": "root", "description": ""}]
    relations = []

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
    ref_target = "entities.json"  # TODO: use correct target

    tax_id = entity_to_id(entity).id_

    if entity["taxonomy_type"] == "tree":
        entities, relations = _parse_tree_node(entity["items"], tax_id)
    elif entity["taxonomy_type"] == "list":
        # gets handled like a tree with depth 1
        entities, relations = _parse_list_to_tree(entity["items"], tax_id)
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

    deduplicated_metadata = {}

    for meta in metadata:
        deduplicated_metadata[meta.ID] = meta

    return deduplicated_metadata


def _main():
    for meta in _entity_class_to_attribute_metadata(PartEntity):
        print(meta)


if __name__ == "__main__":
    _main()
