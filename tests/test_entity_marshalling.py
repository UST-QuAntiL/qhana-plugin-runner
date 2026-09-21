# Copyright 2021 QHAna plugin runner contributors.
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

"""Tests for the entity_marshalling module."""

from collections import namedtuple
from json import loads
from typing import Any, Dict, Iterable, Iterator, TextIO

from hypothesis import given
from hypothesis import strategies as st

from qhana_plugin_runner.plugin_utils.attributes import (
    parse_attribute_metadata,
    tuple_serializer,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    ensure_tuple,
    load_entities,
    save_entities,
)

from .test_entity_attributes import (
    DEFAULT_ATTR_METADATA,
    DEFAULT_ATTRIBUTES,
    DEFAULT_ENTITY_STRATEGY,
    DEFAULT_ENTITY_TUPLE,
    DEFAULT_ENTITY_TUPLE_STRATEGY,
    LIST_ENTITY_ATTRIBUTES,
    LIST_ENTITY_TUPLE,
    LIST_ENTITY_TUPLE_STRATEGY,
    SET_ENTITY_ATTRIBUTES,
    SET_ENTITY_TUPLE,
    SET_ENTITY_TUPLE_STRATEGY,
)
from .utils import assert_sequence_equals, assert_sequence_partial_equals


class ReadWriteDummy(TextIO):
    """Dummy to simulate writing files and reading from them as a response object.

    Inherits from TextIO purely to satisfy type checkers.
    """

    def __init__(self, data: str = "") -> None:
        self.data: str = data

    def json(self, **kwargs):
        return loads(self.data)

    def iter_lines(self, *args, **kwargs) -> Iterator[Any]:
        return iter(self.data.splitlines(keepends=True))

    def write(self, data: str):
        self.data += data

    def writelines(self, lines: Iterable[str]):
        self.data += "".join(lines)


@given(entity_dict=DEFAULT_ENTITY_STRATEGY)
def test_entity_tuple_mixin(entity_dict: Dict[str, Any]):
    entity = DEFAULT_ENTITY_TUPLE.from_dict(**entity_dict)
    assert isinstance(entity, DEFAULT_ENTITY_TUPLE)
    get_by_index = [entity.get(i) for i in range(len(entity))]
    assert all(
        v is not None for v in get_by_index
    ), "subscripting with numbers is not working"
    get_by_key_get = [entity.get(key) for key in DEFAULT_ATTRIBUTES]
    assert all(v is not None for v in get_by_key_get), "get with keys is not working"
    assert_sequence_equals(get_by_index, entity)
    assert_sequence_equals(get_by_key_get, entity)


@given(start=st.lists(DEFAULT_ENTITY_STRATEGY))
def test_ensure_roundtrip(start: list):
    """Test roundtrip from entity dict to tuple and back."""
    end = list(
        ensure_dict(ensure_tuple(start, tuple_=namedtuple("test", DEFAULT_ATTRIBUTES)))
    )
    assert_sequence_equals(expected=start, actual=end)


@given(
    entities=st.lists(DEFAULT_ENTITY_STRATEGY),
    mimetype=st.one_of(st.just("application/json"), st.just("application/X-lines+json")),
)
def test_json_roundtrip(entities: list, mimetype: str):
    """Test json serialization roundtrip."""
    dummy_file = ReadWriteDummy()
    save_entities(entities=entities, file_=dummy_file, mimetype=mimetype)
    read_entities = list(load_entities(file_=dummy_file, mimetype=mimetype))
    assert_sequence_equals(expected=entities, actual=read_entities)


@given(
    entities=st.lists(DEFAULT_ENTITY_TUPLE_STRATEGY),
    mimetype=st.one_of(st.just("application/json"), st.just("application/X-lines+json")),
)
def test_json_tuples(entities: list, mimetype: str):
    """Test json serialization roundtrip from tuples and to tuples."""
    dummy_file = ReadWriteDummy()
    save_entities(entities=entities, file_=dummy_file, mimetype=mimetype)
    read_entities = list(
        ensure_tuple(
            load_entities(file_=dummy_file, mimetype=mimetype),
            tuple_=DEFAULT_ENTITY_TUPLE,
        )
    )
    assert_sequence_equals(expected=entities, actual=read_entities)


@given(
    entities=st.lists(st.one_of(DEFAULT_ENTITY_STRATEGY, DEFAULT_ENTITY_TUPLE_STRATEGY))
)
def test_csv_roundtrip(entities):
    """Test csv serialization roundtrip, reading entities as tuples."""
    mimetype = "text/csv"
    dummy_file = ReadWriteDummy()
    save_entities(
        entities=entities,
        file_=dummy_file,
        mimetype=mimetype,
        attributes=DEFAULT_ATTRIBUTES,
    )
    read_entities = list(load_entities(file_=dummy_file, mimetype=mimetype))
    assert_sequence_partial_equals(
        expected=entities, actual=read_entities, attributes_to_test=["ID", "href"]
    )

    # needs second round trip test as csv converts everything to strings!
    dummy_file_2 = ReadWriteDummy()
    save_entities(
        entities=entities,
        file_=dummy_file_2,
        mimetype=mimetype,
        attributes=DEFAULT_ATTRIBUTES,
    )
    assert dummy_file.data == dummy_file_2.data
    read_entities_2 = list(load_entities(file_=dummy_file_2, mimetype=mimetype))
    assert_sequence_equals(expected=read_entities, actual=read_entities_2)


@given(
    entities=st.lists(st.one_of(DEFAULT_ENTITY_STRATEGY, DEFAULT_ENTITY_TUPLE_STRATEGY))
)
def test_csv_tuples(entities):
    """Test csv serialization roundtrip, reading entities as dicts."""
    mimetype = "text/csv"
    dummy_file = ReadWriteDummy()
    save_entities(
        entities=entities,
        file_=dummy_file,
        mimetype=mimetype,
        attributes=DEFAULT_ATTRIBUTES,
    )
    read_entities = list(ensure_dict(load_entities(file_=dummy_file, mimetype=mimetype)))
    assert_sequence_partial_equals(
        expected=entities, actual=read_entities, attributes_to_test=["ID", "href"]
    )

    # needs second round trip test as csv converts everything to strings!
    dummy_file_2 = ReadWriteDummy()
    save_entities(
        entities=entities,
        file_=dummy_file_2,
        mimetype=mimetype,
        attributes=DEFAULT_ATTRIBUTES,
    )
    assert dummy_file.data == dummy_file_2.data
    read_entities_2 = list(
        ensure_dict(load_entities(file_=dummy_file_2, mimetype=mimetype))
    )
    assert_sequence_equals(expected=read_entities, actual=read_entities_2)


@given(
    entities=st.lists(st.one_of(DEFAULT_ENTITY_STRATEGY, DEFAULT_ENTITY_TUPLE_STRATEGY))
)
def test_ensure_dict_with_metadata_csv_roundtrip(entities):
    """Test that ensure_dict with metadata restores typed values from csv."""
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))
    dummy_file = ReadWriteDummy()
    save_entities(
        entities=entities,
        file_=dummy_file,
        mimetype="text/csv",
        attributes=DEFAULT_ATTRIBUTES,
    )
    read_entities = list(
        ensure_dict(
            load_entities(file_=dummy_file, mimetype="text/csv"),
            attribute_metadata=attr_metadata,
        )
    )
    expected = [
        entity if isinstance(entity, dict) else entity.as_dict() for entity in entities
    ]
    assert_sequence_equals(expected=expected, actual=read_entities)


@given(
    entities=st.lists(st.one_of(DEFAULT_ENTITY_STRATEGY, DEFAULT_ENTITY_TUPLE_STRATEGY))
)
def test_ensure_dict_without_metadata_keeps_strings(entities):
    """Test that ensure_dict without metadata keeps raw csv string values."""
    dummy_file = ReadWriteDummy()
    save_entities(
        entities=entities,
        file_=dummy_file,
        mimetype="text/csv",
        attributes=DEFAULT_ATTRIBUTES,
    )
    read_entities = list(
        ensure_dict(load_entities(file_=dummy_file, mimetype="text/csv"))
    )
    for entity in read_entities:
        for value in entity.values():
            assert isinstance(
                value, str
            ), f"Value {value} of entity {entity} is not a string!"
    assert_sequence_partial_equals(
        expected=entities, actual=read_entities, attributes_to_test=["ID", "href"]
    )


@given(entities=st.lists(DEFAULT_ENTITY_TUPLE_STRATEGY))
def test_ensure_dict_metadata_partial(entities):
    """Test that attributes without a metadata entry keep raw strings."""
    attr_metadata = parse_attribute_metadata(
        ensure_dict(meta for meta in DEFAULT_ATTR_METADATA if meta.ID == "integer")
    )
    dummy_file = ReadWriteDummy()
    save_entities(
        entities=entities,
        file_=dummy_file,
        mimetype="text/csv",
        attributes=DEFAULT_ATTRIBUTES,
    )
    read_entities = list(
        ensure_dict(
            load_entities(file_=dummy_file, mimetype="text/csv"),
            attribute_metadata=attr_metadata,
        )
    )
    assert len(read_entities) == len(entities)
    for expected, actual in zip(entities, read_entities):
        assert actual["integer"] == expected.get("integer")
        assert isinstance(actual["integer"], int)
        assert isinstance(actual["number"], str)
        assert isinstance(actual["boolean"], str)


@given(entities=st.lists(DEFAULT_ENTITY_STRATEGY))
def test_ensure_dict_dict_passthrough_with_metadata(entities):
    """Test that dict entities pass through ensure_dict unchanged."""
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))
    result = list(ensure_dict(entities, attribute_metadata=attr_metadata))
    assert_sequence_equals(expected=entities, actual=result)
    for expected, actual in zip(entities, result):
        assert expected is actual, "Dict entities must be yielded unchanged!"


@given(entities=st.lists(LIST_ENTITY_TUPLE_STRATEGY))
def test_ensure_dict_metadata_list_attributes(entities):
    """Test that ensure_dict deserializes ordered multi attributes to lists."""
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))
    serialize = tuple_serializer(
        LIST_ENTITY_ATTRIBUTES, attr_metadata, tuple_=LIST_ENTITY_TUPLE._make
    )
    serialized_entities = [serialize(entity) for entity in entities]
    read_entities = list(
        ensure_dict(serialized_entities, attribute_metadata=attr_metadata)
    )
    expected = [entity.as_dict() for entity in entities]
    assert_sequence_equals(expected=expected, actual=read_entities)


@given(entities=st.lists(SET_ENTITY_TUPLE_STRATEGY))
def test_ensure_dict_metadata_set_attributes(entities):
    """Test that ensure_dict deserializes unordered multi attributes to sets."""
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))
    serialize = tuple_serializer(
        SET_ENTITY_ATTRIBUTES, attr_metadata, tuple_=SET_ENTITY_TUPLE._make
    )
    serialized_entities = [serialize(entity) for entity in entities]
    read_entities = list(
        ensure_dict(serialized_entities, attribute_metadata=attr_metadata)
    )
    expected = [entity.as_dict() for entity in entities]
    assert_sequence_equals(expected=expected, actual=read_entities)


@given(entities=st.lists(DEFAULT_ENTITY_TUPLE_STRATEGY))
def test_ensure_dict_plain_namedtuple_ignores_metadata(entities):
    """Test that plain namedtuples (no entity tuple class) stay raw."""
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))
    plain_tuple = namedtuple("PlainEntityTuple", DEFAULT_ATTRIBUTES)
    plain_entities = [
        plain_tuple(*(str(value) for value in entity)) for entity in entities
    ]
    result = list(ensure_dict(plain_entities, attribute_metadata=attr_metadata))
    assert len(result) == len(entities)
    for entity in result:
        assert set(entity.keys()) == set(DEFAULT_ATTRIBUTES)
        for value in entity.values():
            assert isinstance(
                value, str
            ), f"Value {value} of entity {entity} is not a string!"
