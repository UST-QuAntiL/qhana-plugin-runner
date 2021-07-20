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

from hypothesis import given
from hypothesis import strategies as st
from test_entity_marshalling import (
    DEFAULT_ATTRIBUTES,
    DEFAULT_ENTITY_STRATEGY,
    DEFAULT_ENTITY_TUPLE,
    DEFAULT_ENTITY_TUPLE_STRATEGY,
)
from utils import assert_sequence_equals, assert_sequence_partial_equals

from qhana_plugin_runner.plugin_utils.attributes import (
    AttributeMetadata,
    dict_deserializer,
    dict_serializer,
    parse_attribute_metadata,
    tuple_deserializer,
    tuple_serializer,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import ensure_dict

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
    deserialized_entities = list(deserialize(entity) for entity in entities)
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
    deserialized_entities = list(deserialize(entity) for entity in entities)
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
    deserialized_entities = list(deserialize(entity) for entity in entities)
    assert_sequence_equals(expected=entities, actual=deserialized_entities)
