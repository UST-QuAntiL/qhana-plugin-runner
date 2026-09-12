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

"""Tests for the pure numeric attribute helpers (no worker, no database)."""

import json
from io import BytesIO
from zipfile import ZipFile

import pytest

from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from feature_engineering_pipeline.numeric_attributes import (
    collect_values,
    entities_zip,
    normalized_column,
    parse_single_value,
    parse_vector,
)


def _metadata(multiple: bool) -> AttributeMetadata:
    return AttributeMetadata(
        ID="attr",
        attribute_type="attr",
        title="attr",
        description="integer",
        multiple=multiple,
        separator=";",
    )


# --- SINGLE VALUES ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3", 3.0),
        (" 2.5 ", 2.5),
        (7, 7.0),
        (1.5, 1.5),
        ("-1", -1.0),
        (None, None),
        ("", None),
        ("   ", None),
        ("nan", None),
        ("NaN", None),
        (float("nan"), None),
    ],
)
def test_parse_single_value(raw, expected):
    assert parse_single_value("e1", "attr", raw) == expected


@pytest.mark.parametrize("raw", [[1, 2], (1, 2), "abc", "1;2", True, {"a": 1}])
def test_parse_single_value_rejects_non_numeric_values(raw):
    with pytest.raises(ValueError, match="'e1'.*'attr'"):
        parse_single_value("e1", "attr", raw)


# --- VECTORS ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1;2;3", [1.0, 2.0, 3.0]),
        ("1; 2 ;3", [1.0, 2.0, 3.0]),
        ([1, 2, 3], [1.0, 2.0, 3.0]),
        ((4, 5), [4.0, 5.0]),
        (2, [2.0]),
        ("1;;3", [1.0, None, 3.0]),
        ("1;nan;3", [1.0, None, 3.0]),
        (None, None),
        ("", None),
        ("  ", None),
        (";", None),
        (";;", None),
        ([], None),
        ([None, ""], None),
    ],
)
def test_parse_vector(raw, expected):
    assert parse_vector("e1", "attr", raw, ";") == expected


def test_parse_vector_uses_the_separator_of_the_attribute():
    assert parse_vector("e1", "attr", "1|2", "|") == [1.0, 2.0]


@pytest.mark.parametrize("raw", ["1;x;3", ["1", "y"], [[1, 2]], "1,2"])
def test_parse_vector_rejects_non_numeric_parts(raw):
    with pytest.raises(ValueError, match="'e1'.*'attr'"):
        parse_vector("e1", "attr", raw, ";")


# --- COLLECTING VALUES ---


def test_collect_values_keeps_the_entity_order():
    entities = [
        {"ID": "b", "attr": "2"},
        {"ID": "a", "attr": ""},
        {"ID": "c", "attr": "1"},
    ]
    assert collect_values(entities, "attr", _metadata(multiple=False)) == [
        2.0,
        None,
        1.0,
    ]


def test_collect_values_parses_vectors_for_multi_valued_attributes():
    entities = [{"ID": "a", "attr": "1;2"}, {"ID": "b", "attr": None}]
    assert collect_values(entities, "attr", _metadata(multiple=True)) == [
        [1.0, 2.0],
        None,
    ]


def test_collect_values_treats_a_missing_column_as_missing_values():
    assert collect_values([{"ID": "a"}], "attr", _metadata(multiple=False)) == [None]


def test_collect_values_rejects_vectors_of_different_lengths():
    entities = [{"ID": "a", "attr": "1;2"}, {"ID": "b", "attr": "1;2;3"}]
    with pytest.raises(ValueError, match="different lengths"):
        collect_values(entities, "attr", _metadata(multiple=True))


# --- NORMALIZATION ---


def test_normalized_column_scales_to_unit_interval():
    assert normalized_column([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]


def test_normalized_column_fills_missing_values_with_the_mean():
    # mean of the known values 0 and 10 is 5, which scales to 0.5
    assert normalized_column([0.0, None, 10.0]) == [0.0, 0.5, 1.0]


def test_normalized_column_maps_a_constant_attribute_to_zero():
    assert normalized_column([3.0, 3.0, None]) == [0.0, 0.0, 0.0]


def test_normalized_column_rejects_a_column_without_values():
    with pytest.raises(ValueError, match="no numeric value"):
        normalized_column([None, None])


# --- ZIP OUTPUT ---


def test_entities_zip_writes_one_json_member_per_attribute():
    members = {
        "a": [{"ID": "e1", "href": "", "dim0": 0.5}],
        "b": [{"source": "e1", "target": "e2", "distance": None}],
    }

    # ZipFile cannot read a SpooledTemporaryFile on Python 3.10 (no seekable)
    with ZipFile(BytesIO(entities_zip(members).read())) as archive:
        assert sorted(archive.namelist()) == ["a.json", "b.json"]
        assert json.loads(archive.read("a.json")) == members["a"]
        assert json.loads(archive.read("b.json")) == members["b"]
