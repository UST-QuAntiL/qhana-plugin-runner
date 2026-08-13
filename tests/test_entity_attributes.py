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

"""Tests for the attributes module of the plugin_utils."""

from collections import namedtuple
from typing import NamedTuple, Type

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qhana_plugin_runner.plugin_utils.attributes import (
    AttributeMetadata,
    dict_deserializer,
    dict_serializer,
    parse_attribute_metadata,
    tuple_deserializer,
    tuple_serializer,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    get_entity_tuple_class,
)

from .utils import assert_sequence_equals, assert_sequence_partial_equals

CSV_UNSAFE_CHARACTERS = ["\x00"]

DEFAULT_ATTRIBUTES = ["ID", "href", "integer", "number", "boolean"]

DEFAULT_ENTITY_TUPLE: Type[NamedTuple] = get_entity_tuple_class(
    DEFAULT_ATTRIBUTES, name="DefaultEntityTuple"
)

DEFAULT_ENTITY_STRATEGY = st.fixed_dictionaries(
    {
        "ID": st.text(st.characters(blacklist_characters=CSV_UNSAFE_CHARACTERS)),
        "href": st.text(st.characters(blacklist_characters=CSV_UNSAFE_CHARACTERS)),
        "integer": st.integers(),
        "number": st.floats(allow_infinity=False, allow_nan=False),
        "boolean": st.booleans(),
    }
)

DEFAULT_ENTITY_TUPLE_STRATEGY = st.builds(
    DEFAULT_ENTITY_TUPLE,
    ID=st.text(st.characters(blacklist_characters=CSV_UNSAFE_CHARACTERS)),
    href=st.text(st.characters(blacklist_characters=CSV_UNSAFE_CHARACTERS)),
    integer=st.integers(),
    number=st.floats(allow_infinity=False, allow_nan=False),
    boolean=st.booleans(),
)

ATTR_METADATA_TUPLE = namedtuple(
    "AttributeMetadataTuple",
    [
        "ID",
        "type",
        "title",
        "description",
        "multiple",
        "ordered",
        "separator",
        "refTarget",
        "schema",
    ],
)


DEFAULT_ATTR_METADATA = [
    ATTR_METADATA_TUPLE("ID", "string", "Entity ID", "", False, False, ";", None, None),
    ATTR_METADATA_TUPLE(
        "href", "string", "Entity URL", "", False, False, ";", None, None
    ),
    ATTR_METADATA_TUPLE(
        "integer", "integer", "Integer Attribute", "", False, False, ";", None, None
    ),
    ATTR_METADATA_TUPLE(
        "number", "double", "Number Attribute", "", False, False, ";", None, None
    ),
    ATTR_METADATA_TUPLE(
        "boolean", "boolean", "Boolean Attribute", "", False, False, ";", None, None
    ),
    ATTR_METADATA_TUPLE(
        "str_list", "string", "String List Attribute", "", True, True, ";", None, None
    ),
    ATTR_METADATA_TUPLE(
        "integer_list",
        "integer",
        "Integer List Attribute",
        "",
        True,
        True,
        ";",
        None,
        None,
    ),
    ATTR_METADATA_TUPLE(
        "number_list", "double", "Number List Attribute", "", True, True, ";", None, None
    ),
    ATTR_METADATA_TUPLE(
        "boolean_list",
        "boolean",
        "Boolean List Attribute",
        "",
        True,
        True,
        ";",
        None,
        None,
    ),
    ATTR_METADATA_TUPLE(
        "str_set", "string", "String Set Attribute", "", True, False, ";", None, None
    ),
    ATTR_METADATA_TUPLE(
        "integer_set",
        "integer",
        "Integer Set Attribute",
        "",
        True,
        False,
        ";",
        None,
        None,
    ),
    ATTR_METADATA_TUPLE(
        "number_set", "double", "Number Set Attribute", "", True, False, ";", None, None
    ),
    ATTR_METADATA_TUPLE(
        "boolean_set",
        "boolean",
        "Boolean Set Attribute",
        "",
        True,
        False,
        ";",
        None,
        None,
    ),
]


@given(entities=st.lists(DEFAULT_ENTITY_TUPLE_STRATEGY))
def test_tuple_serialization_roundtrip(entities: list):
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))

    # serialize
    serialize = tuple_serializer(
        DEFAULT_ATTRIBUTES, attr_metadata, tuple_=DEFAULT_ENTITY_TUPLE._make
    )
    serialized_entities = list(serialize(entity) for entity in entities)
    assert_sequence_partial_equals(
        expected=entities, actual=serialized_entities, attributes_to_test=["ID", "href"]
    )

    # assert all serialized
    for ent in serialized_entities:
        for value in ent:
            assert isinstance(
                value, str
            ), f"Value {value} of entity {ent} did not get serialized correctly!"

    # deserialize
    deserialize = tuple_deserializer(
        DEFAULT_ATTRIBUTES, attr_metadata, tuple_=DEFAULT_ENTITY_TUPLE._make
    )
    deserialized_entities = list(deserialize(entity) for entity in serialized_entities)
    assert_sequence_equals(expected=entities, actual=deserialized_entities)


@given(entities=st.lists(DEFAULT_ENTITY_STRATEGY))
def test_dict_serialization_roundtrip(entities: list):
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))

    # serialize
    serialize = dict_serializer(DEFAULT_ATTRIBUTES, attr_metadata, in_place=False)
    serialized_entities = list(serialize(entity) for entity in entities)
    assert_sequence_partial_equals(
        expected=entities, actual=serialized_entities, attributes_to_test=["ID", "href"]
    )

    # assert all serialized
    for ent in serialized_entities:
        for value in ent:
            assert isinstance(
                value, str
            ), f"Value {value} of entity {ent} did not get serialized correctly!"

    # deserialize
    deserialize = dict_deserializer(DEFAULT_ATTRIBUTES, attr_metadata, in_place=False)
    deserialized_entities = list(deserialize(entity) for entity in serialized_entities)
    assert_sequence_equals(expected=entities, actual=deserialized_entities)


@given(entities=st.lists(DEFAULT_ENTITY_STRATEGY))
def test_dict_serialization_roundtrip_in_place(entities: list):
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))

    # serialize
    serialize = dict_serializer(DEFAULT_ATTRIBUTES, attr_metadata, in_place=True)
    serialized_entities = list(serialize(dict(entity)) for entity in entities)
    assert_sequence_partial_equals(
        expected=entities, actual=serialized_entities, attributes_to_test=["ID", "href"]
    )

    # assert all serialized
    for ent in serialized_entities:
        for value in ent:
            assert isinstance(
                value, str
            ), f"Value {value} of entity {ent} did not get serialized correctly!"

    # deserialize
    deserialize = dict_deserializer(DEFAULT_ATTRIBUTES, attr_metadata, in_place=True)
    deserialized_entities = list(deserialize(entity) for entity in serialized_entities)
    assert_sequence_equals(expected=entities, actual=deserialized_entities)


LIST_ENTITY_ATTRIBUTES = ["ID", "str_list", "integer_list", "number_list", "boolean_list"]

LIST_ENTITY_VALUE_STRATEGIES = {
    "ID": st.text(st.characters(blacklist_characters=CSV_UNSAFE_CHARACTERS)),
    "str_list": st.lists(st.text(st.characters(blacklist_characters=[";"]), min_size=1)),
    "integer_list": st.lists(st.integers()),
    "number_list": st.lists(st.floats(allow_infinity=False, allow_nan=False)),
    "boolean_list": st.lists(st.booleans()),
}

LIST_ENTITY_STRATEGY = st.fixed_dictionaries(LIST_ENTITY_VALUE_STRATEGIES)

LIST_ENTITY_TUPLE = get_entity_tuple_class(LIST_ENTITY_ATTRIBUTES, name="ListEntityTuple")

LIST_ENTITY_TUPLE_STRATEGY = st.builds(LIST_ENTITY_TUPLE, **LIST_ENTITY_VALUE_STRATEGIES)


@given(entities=st.lists(LIST_ENTITY_STRATEGY))
def test_list_serialization_roundtrip(entities: list):
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))

    # serialize
    serialize = dict_serializer(LIST_ENTITY_ATTRIBUTES, attr_metadata, in_place=False)
    serialized_entities = list(serialize(entity) for entity in entities)
    assert_sequence_partial_equals(
        expected=entities,
        actual=serialized_entities,
        attributes_to_test=[
            "ID",
        ],
    )

    # assert all serialized
    for ent in serialized_entities:
        for value in ent:
            assert isinstance(
                value, str
            ), f"Value {value} of entity {ent} did not get serialized correctly!"

    # deserialize
    deserialize = dict_deserializer(LIST_ENTITY_ATTRIBUTES, attr_metadata, in_place=False)
    deserialized_entities = list(deserialize(entity) for entity in serialized_entities)
    assert_sequence_equals(expected=entities, actual=deserialized_entities)


@given(entities=st.lists(LIST_ENTITY_TUPLE_STRATEGY))
def test_list_tuple_serialization_roundtrip(entities: list):
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))

    # serialize
    serialize = tuple_serializer(
        LIST_ENTITY_ATTRIBUTES, attr_metadata, tuple_=LIST_ENTITY_TUPLE._make
    )
    serialized_entities = list(serialize(entity) for entity in entities)
    assert_sequence_partial_equals(
        expected=entities,
        actual=serialized_entities,
        attributes_to_test=[
            "ID",
        ],
    )

    # assert all serialized
    for ent in serialized_entities:
        for value in ent:
            assert isinstance(
                value, str
            ), f"Value {value} of entity {ent} did not get serialized correctly!"

    # deserialize
    deserialize = tuple_deserializer(
        LIST_ENTITY_ATTRIBUTES, attr_metadata, tuple_=LIST_ENTITY_TUPLE._make
    )
    deserialized_entities = list(deserialize(entity) for entity in serialized_entities)
    assert_sequence_equals(expected=entities, actual=deserialized_entities)


SET_ENTITY_ATTRIBUTES = ["ID", "str_set", "integer_set", "number_set", "boolean_set"]

SET_ENTITY_VALUE_STRATEGIES = {
    "ID": st.text(st.characters(blacklist_characters=CSV_UNSAFE_CHARACTERS)),
    "str_set": st.sets(st.text(st.characters(blacklist_characters=[";"]), min_size=1)),
    "integer_set": st.sets(st.integers()),
    "number_set": st.sets(st.floats(allow_infinity=False, allow_nan=False)),
    "boolean_set": st.sets(st.booleans()),
}

SET_ENTITY_STRATEGY = st.fixed_dictionaries(SET_ENTITY_VALUE_STRATEGIES)

SET_ENTITY_TUPLE = get_entity_tuple_class(SET_ENTITY_ATTRIBUTES, name="SetEntityTuple")

SET_ENTITY_TUPLE_STRATEGY = st.builds(SET_ENTITY_TUPLE, **SET_ENTITY_VALUE_STRATEGIES)


@given(entities=st.lists(SET_ENTITY_STRATEGY))
def test_set_serialization_roundtrip(entities: list):
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))

    # serialize
    serialize = dict_serializer(SET_ENTITY_ATTRIBUTES, attr_metadata, in_place=False)
    serialized_entities = list(serialize(entity) for entity in entities)
    assert_sequence_partial_equals(
        expected=entities,
        actual=serialized_entities,
        attributes_to_test=[
            "ID",
        ],
    )

    # assert all serialized
    for ent in serialized_entities:
        for value in ent:
            assert isinstance(
                value, str
            ), f"Value {value} of entity {ent} did not get serialized correctly!"

    # deserialize
    deserialize = dict_deserializer(SET_ENTITY_ATTRIBUTES, attr_metadata, in_place=False)
    deserialized_entities = list(deserialize(entity) for entity in serialized_entities)
    assert_sequence_equals(expected=entities, actual=deserialized_entities)


@given(entities=st.lists(SET_ENTITY_TUPLE_STRATEGY))
def test_set_tuple_serialization_roundtrip(entities: list):
    attr_metadata = parse_attribute_metadata(ensure_dict(DEFAULT_ATTR_METADATA))

    # serialize
    serialize = tuple_serializer(
        SET_ENTITY_ATTRIBUTES, attr_metadata, tuple_=SET_ENTITY_TUPLE._make
    )
    serialized_entities = list(serialize(entity) for entity in entities)
    assert_sequence_partial_equals(
        expected=entities,
        actual=serialized_entities,
        attributes_to_test=[
            "ID",
        ],
    )

    # assert all serialized
    for ent in serialized_entities:
        for value in ent:
            assert isinstance(
                value, str
            ), f"Value {value} of entity {ent} did not get serialized correctly!"

    # deserialize
    deserialize = tuple_deserializer(
        SET_ENTITY_ATTRIBUTES, attr_metadata, tuple_=SET_ENTITY_TUPLE._make
    )
    deserialized_entities = list(deserialize(entity) for entity in serialized_entities)
    assert_sequence_equals(expected=entities, actual=deserialized_entities)


@pytest.mark.parametrize(
    "separator",
    [
        pytest.param(..., id="missing"),
        pytest.param(None, id="null"),
        pytest.param("", id="empty"),
    ],
)
def test_multiple_without_separator_raises(separator):
    metadata = {"ID": "x", "type": "string", "multiple": True}
    if separator is not ...:
        metadata["separator"] = separator

    with pytest.raises(ValueError, match="'x'"):
        AttributeMetadata.from_dict(metadata)

    kwargs = {} if separator is ... else {"separator": separator}

    with pytest.raises(ValueError, match="'x'"):
        AttributeMetadata("x", "string", "X", multiple=True, **kwargs)


@pytest.mark.parametrize(
    "separator",
    [
        pytest.param(" ", id="space"),
        pytest.param("\n", id="newline"),
        pytest.param(";", id=";"),
        pytest.param("$", id="$"),
    ],
)
def test_multiple_with_separator(separator):
    attr = AttributeMetadata.from_dict(
        {"ID": "x", "type": "string", "multiple": True, "separator": separator}
    )
    assert attr.multiple is True
    assert attr.separator == separator

    attr2 = AttributeMetadata("y", "string", "Y", multiple=True, separator=separator)
    assert attr2.multiple is True
    assert attr2.separator == separator
