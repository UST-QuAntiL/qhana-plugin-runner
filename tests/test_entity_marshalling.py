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
from keyword import iskeyword
from typing import Any, Iterable, Iterator, Sequence, TextIO

from hypothesis import given
from hypothesis import strategies as st
from utils import assert_sequence_equals, assert_sequence_partial_equals

from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    ensure_tuple,
    load_entities,
    normalize_attribute_name,
    save_entities,
)

CSV_UNSAFE_CHARACTERS = ["\x00"]

DEFAULT_ATTRIBUTES = ["ID", "href", "integer", "number", "boolean"]

DEFAULT_ENTITY_TUPLE = namedtuple("DefaultEntityTuple", DEFAULT_ATTRIBUTES)

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


@given(name=st.text(min_size=1))
def test_normalize_attribute_name(name):
    """Test the attribute name normalization."""
    normalized = normalize_attribute_name(name)
    assert (
        normalized.isidentifier()
    ), f'Attribute "{name}" was not normalized to a valid identifier! (Normalized: "{normalized}")'
    assert not iskeyword(
        normalized
    ), f'Attribute "{name}" was normalized to a python keyword!'


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
