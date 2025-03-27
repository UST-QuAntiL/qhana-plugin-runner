# Copyright 2025 QHAna plugin runner contributors.
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

from json import loads
from tempfile import SpooledTemporaryFile
from typing import Literal, Optional, Sequence, Union

from celery.utils.log import get_task_logger
from requests.exceptions import ConnectionError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import (
    AttributeMetadata,
    dict_deserializer,
    dict_serializer,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.requests import (
    get_mimetype,
    open_url,
    retrieve_attribute_metadata_url,
    retrieve_data_type,
    retrieve_filename,
)
from qhana_plugin_runner.storage import STORE

from . import DataJoin

TASK_LOGGER = get_task_logger(__name__)


def _get_file_metadata(entity_url: str) -> dict:
    with open_url(entity_url) as response:
        first_entity = next(
            ensure_dict(
                load_entities(response, get_mimetype(response)),
            )
        )

        metadata = {
            "data": entity_url,
            "attribute_metadata": retrieve_attribute_metadata_url(response),
            "name": retrieve_filename(response),
            "data_type": retrieve_data_type(response),
            "content_type": get_mimetype(response),
            "attributes": list(first_entity.keys()),
        }

        return metadata


@CELERY.task(name=f"{DataJoin.instance.identifier}.load_base", bind=True)
def load_base(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Load info from the base entity file for job '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = loads(task_data.parameters)

    data = task_data.data
    assert data is not None
    assert isinstance(data, dict)

    base_data_url = params["base"]
    metadata = _get_file_metadata(base_data_url)
    data["attributes"] = metadata.pop("attributes")
    data["base"] = metadata

    task_data.save(commit=True)

    if not data["attributes"]:
        raise ValueError("Base entities doe not have any attributes!")

    return "Loaded metadata from base data finished."


@CELERY.task(name=f"{DataJoin.instance.identifier}.add_data_to_join", bind=True)
def add_data_to_join(self, db_id: int, entity_url: str, join_attr: str):
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to update data!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    data = task_data.data
    if not data:
        data = {}
        task_data.data = data
    assert isinstance(data, dict)

    attributes = data.get("attributes")
    assert isinstance(attributes, (list, tuple, set))
    if join_attr not in attributes:
        msg = f"Cannot join to a nonexistant attribute {join_attr}! (Attributes: {attributes})"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    try:
        file_metadata = _get_file_metadata(entity_url)
        file_metadata["join_attr"] = join_attr

        joins = data.setdefault("joins", [])
        assert isinstance(joins, list)
        joins.append(file_metadata)

        task_data.save(commit=True)
    except ConnectionError:
        return f"Could not load entities from {entity_url}!"
    # TODO catch more errors gracefully?

    return "Finished adding data to join!"


def _is_vector_type(data_type):
    return data_type in ("entity/vector", "entity/matrix")


def _extract_final_attrs(
    attrs_to_keep: Sequence[str],
    entitiy_attrs: Sequence[str],
    processed_final_attrs: tuple[str, ...],
    is_vector_type: bool = False,
) -> tuple[tuple[str, ...], dict[str, str], set[str]]:
    seen_names = set(processed_final_attrs)
    new_attrs: list[str] = []
    vector_attrs: list[str] = []
    replacements: dict[str, str] = {}
    for attr in attrs_to_keep:
        if attr not in entitiy_attrs:
            raise KeyError(
                f"Attribute {attr} was selected, but is not avaliable. (Available attrs: {entitiy_attrs})"
            )
        unique_attr = attr
        counter = 0
        while unique_attr in seen_names:
            counter += 1
            unique_attr = f"{attr}_{counter}"
        seen_names.add(unique_attr)
        replacements[attr] = unique_attr
        new_attrs.append(unique_attr)
        if attr in ("ID", "href"):
            continue
        if is_vector_type:
            vector_attrs.append(unique_attr)
    return processed_final_attrs + tuple(new_attrs), replacements, set(vector_attrs)


def _sort_final_attrs(
    final_attrs: tuple[str, ...],
    vector_attrs: set[str],
    attr_replacements: dict[Union[Literal["base"], int], dict[str, str]],
) -> tuple[str, ...]:
    dimension = len(vector_attrs)
    dimension_len = len(str(dimension))
    sorted_vector_attrs = [a for a in final_attrs if a in vector_attrs]
    filtered_final_attrs = tuple(a for a in final_attrs if a not in vector_attrs)

    # rename all vector attrs in lxicographical order
    prefix = "dim_"
    suffix = ""
    count = 0
    while any(a.startswith(prefix) and a.endswith(suffix) for a in filtered_final_attrs):
        count += 1
        suffix = f"_{count}"
    renamed_vector_attrs = [
        f"{prefix}{index:0{dimension_len}}{suffix}" for index in range(dimension)
    ]

    # adjust all existing replacement dicts to reflect new renaming
    vector_attr_replacements = dict(zip(sorted_vector_attrs, renamed_vector_attrs))
    for replacement_dict in attr_replacements.values():
        for attr in replacement_dict:
            if replacement_dict[attr] in vector_attr_replacements:
                replacement_dict[attr] = vector_attr_replacements[replacement_dict[attr]]

    # place vector attrs last
    return filtered_final_attrs + tuple(renamed_vector_attrs)


def _load_attribute_metadata(entity_data: dict):
    entities_metadata_url = entity_data.get("attribute_metadata")
    if not entities_metadata_url:
        return {}
    entities_metadata: dict[str, AttributeMetadata]
    with open_url(entities_metadata_url) as response:
        metadata = ensure_dict(load_entities(response, get_mimetype(response)))
        entities_metadata = {m["ID"]: AttributeMetadata.from_dict(m) for m in metadata}
    return entities_metadata


def _load_entities(
    response,
    attributes: Sequence[str],
    entity_data: dict,
    attribute_metadata: dict[str, AttributeMetadata],
):
    content_type = get_mimetype(response, entity_data["content_type"])
    assert content_type
    deserialize = dict_deserializer(attributes, attribute_metadata)
    return (deserialize(e) for e in ensure_dict(load_entities(response, content_type)))


def _determine_datatype(
    base_metadata: dict,
    join_metadata: list[dict],
    used_attributes: dict[Union[Literal["base"], int], dict[str, str]],
):
    data_types: set[str] = set()

    if set(used_attributes.values()) > {"ID", "href"}:
        data_types.add(base_metadata["data_type"])

    for i, join in enumerate(join_metadata):
        if set(used_attributes[i].values()) > {"ID", "href"}:
            data_types.add(join["data_type"])

    for data_type in (
        "entity/label",
        "entity/shaped_vector",
        "entity/matrix",
        "entity/vector",
        "entity/numeric",
        "entity/stream",
    ):
        if len(data_types) == 1:
            return data_types.pop()
        data_types.discard(data_type)
    return "entity/list"


@CELERY.task(name=f"{DataJoin.instance.identifier}.join_data", bind=True)
def join_data(self, db_id: int):  # noqa: C901
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to update data!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    data = task_data.data
    assert isinstance(data, dict)

    params: dict[str, list[str]] = loads(task_data.parameters)

    # process attributes #######################################################
    final_attributes: tuple[str, ...] = tuple()
    vector_attrs: set[str] = set()
    attr_replacements: dict[Union[Literal["base"], int], dict[str, str]] = {}

    base = data["base"]
    assert isinstance(base, (dict))
    base_attrs = data["attributes"]
    assert isinstance(base_attrs, (list, tuple, set))

    final_attributes, base_replacements, vector_attrs = _extract_final_attrs(
        params["base"],
        base_attrs,
        final_attributes,
        _is_vector_type(base.get("data_type")),
    )
    attr_replacements["base"] = base_replacements

    joins = data["joins"]
    assert isinstance(joins, (list, tuple))

    for i, join in enumerate(joins):
        assert isinstance(join, dict)
        attrs_to_keep = params[f"join_{i+1}"]
        final_attributes, join_replacements, new_vector_attrs = _extract_final_attrs(
            params["base"],
            attrs_to_keep,
            final_attributes,
            _is_vector_type(join.get("data_type")),
        )
        attr_replacements[i] = join_replacements
        vector_attrs.update(new_vector_attrs)

    final_attributes = _sort_final_attrs(
        final_attributes, vector_attrs, attr_replacements
    )

    # create join base #########################################################
    final_attr_metadata: list[dict] = []
    joined_entities: dict[str, dict] = {}
    join_maps: dict[str, dict[str, set[str]]] = {j["join_attr"]: {} for j in joins}

    base_metadata: dict[str, AttributeMetadata] = _load_attribute_metadata(base)

    replacements = attr_replacements["base"]

    # process base metadata
    for attr, final_attr in replacements.items():
        metadata_dict = base_metadata[attr].to_dict()
        metadata_dict["ID"] = final_attr  # rename attribute for final attr metadata
        final_attr_metadata.append(metadata_dict)

    # process base entities
    with open_url(base["data"]) as response:
        for entity in _load_entities(response, base_attrs, base, base_metadata):
            id_ = entity["ID"]
            base_entity = {a: None for a in final_attributes}
            base_entity["ID"] = id_
            joined_entities[id_] = base_entity

            for attr, final_attr in replacements.items():
                base_entity[final_attr] = entity[attr]

            for attr, join_map in join_maps.items():
                value = entity[attr]
                join_map.setdefault(value, set()).add(id_)

    for i, join in joins:
        join_attrs = join["attributes"]
        join_metadata = _load_attribute_metadata(join)

        replacements = attr_replacements[i]

        # process joined metadata
        for attr, final_attr in replacements.items():
            metadata_dict = join_metadata[attr].to_dict()
            metadata_dict["ID"] = final_attr  # rename attribute for final attr metadata
            final_attr_metadata.append(metadata_dict)

        join_map = join_maps[join["join_attr"]]

        # process joined entities
        with open_url(join["data"]) as response:
            for entity in _load_entities(response, join_attrs, join, join_metadata):
                id_ = entity["ID"]

                for base_id in join_map.get(id_, []):
                    base_entity = joined_entities[base_id]

                    for attr, final_attr in replacements.items():
                        base_entity[final_attr] = entity[attr]

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                final_attr_metadata,
                output,
                "application/json",
            )
            STORE.persist_task_result(
                db_id,
                output,
                "attribute_metadata.json",
                "entity/attribute-metadata",
                "application/json",
            )

        final_attr_metadata_dict = {
            m["ID"]: AttributeMetadata.from_dict(m) for m in final_attr_metadata
        }
        final_entities_serializer = dict_serializer(
            final_attributes, final_attr_metadata_dict
        )

        final_filename = base["name"]
        final_content_type = base["content_type"]
        final_data_type = _determine_datatype(base, joins, attr_replacements)
        if (
            final_data_type == "entity/stream"
            and final_content_type == "application/json"
        ):
            final_data_type = "entity/list"

        with SpooledTemporaryFile(mode="w") as output:
            save_entities(
                (final_entities_serializer(e) for e in joined_entities.values()),
                output,
                final_content_type,
            )
            STORE.persist_task_result(
                db_id,
                output,
                final_filename,
                final_data_type,
                final_content_type,
            )

    return "Finished joining data!"
