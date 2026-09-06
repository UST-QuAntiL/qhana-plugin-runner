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

"""Tests for the dimension_mapping module."""

import json
from collections import namedtuple

import pytest

from qhana_plugin_runner.plugin_utils.dimension_mapping import (
    dimension_labels,
    dimension_mapping_labels,
    entity_dimension_names,
    load_dimension_mapping,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import ensure_array

from .utils import MockResponse

MAPPING_URL = "http://example.com/concatenated_dimension_mapping.json"


def _mapping_entity(dimension, source, source_dimension, zip_member="", url="url"):
    return {
        "ID": dimension,
        "href": "",
        "inputIndex": 0,
        "source": source,
        "sourceUrl": url,
        "zipMember": zip_member,
        "sourceDimension": source_dimension,
    }


def _serve(monkeypatch, mapping):
    response = MockResponse(MAPPING_URL, "application/json", json_data=mapping)
    monkeypatch.setattr(
        "qhana_plugin_runner.plugin_utils.dimension_mapping.open_url",
        lambda url, *args, **kwargs: response,
    )


def test_label_includes_source_dimension_for_multi_dimensional_sources():
    labels = dimension_mapping_labels(
        [
            _mapping_entity("dim0", "color", "dim0", url="color-url"),
            _mapping_entity("dim1", "color", "dim1", url="color-url"),
            _mapping_entity("dim2", "shape", "dim0", url="shape-url"),
        ]
    )

    assert labels == {
        "dim0": "color (dim0)",
        "dim1": "color (dim1)",
        "dim2": "shape",
    }


def test_label_keeps_original_column_names():
    labels = dimension_mapping_labels(
        [
            _mapping_entity("dim0", "points", "x"),
            _mapping_entity("dim1", "points", "y"),
        ]
    )

    assert labels == {"dim0": "points (x)", "dim1": "points (y)"}


def test_zip_members_and_plain_urls_yield_the_same_name():
    from_zip = dimension_mapping_labels(
        [_mapping_entity("dim0", "color.json", "dim0", zip_member="color.json")]
    )
    from_url = dimension_mapping_labels([_mapping_entity("dim0", "color", "dim0")])

    assert from_zip == from_url == {"dim0": "color"}


def test_sources_are_distinguished_by_url_and_zip_member():
    labels = dimension_mapping_labels(
        [
            _mapping_entity("dim0", "color", "dim0", url="first"),
            _mapping_entity("dim1", "color", "dim0", url="second"),
        ]
    )

    assert labels == {"dim0": "color", "dim1": "color"}


@pytest.mark.parametrize("source", [None, "", "  "])
def test_entities_without_a_source_are_skipped(source):
    labels = dimension_mapping_labels(
        [
            _mapping_entity("dim0", source, "dim0"),
            _mapping_entity("dim1", "color", "dim0"),
        ]
    )

    assert labels == {"dim1": "color"}


def test_entities_without_a_dimension_name_are_skipped():
    assert dimension_mapping_labels([_mapping_entity("", "color", "dim0")]) == {}


def test_missing_source_dimension_falls_back_to_the_source_name():
    labels = dimension_mapping_labels(
        [
            _mapping_entity("dim0", "color", ""),
            _mapping_entity("dim1", "color", "dim1"),
        ]
    )

    assert labels == {"dim0": "color", "dim1": "color (dim1)"}


def test_dimension_labels_fall_back_to_the_dimension_name():
    assert dimension_labels(["dim0", "dim1"], {"dim0": "color"}) == ["color", "dim1"]


@pytest.mark.parametrize("url", [None, ""])
def test_no_url_yields_no_labels(url):
    assert load_dimension_mapping(url) == {}


def test_load_dimension_mapping_reads_the_file(monkeypatch):
    _serve(
        monkeypatch,
        [
            _mapping_entity("dim0", "color", "dim0", url="color-url"),
            _mapping_entity("dim1", "color", "dim1", url="color-url"),
            _mapping_entity("dim2", "shape", "dim0", url="shape-url"),
        ],
    )

    assert load_dimension_mapping(MAPPING_URL) == {
        "dim0": "color (dim0)",
        "dim1": "color (dim1)",
        "dim2": "shape",
    }


def test_load_dimension_mapping_without_mimetype_raises(monkeypatch):
    # A url without a file extension gives no mimetype hint.
    response = MockResponse("http://example.com/download", "", json_data=[])
    del response.headers["Content-Type"]
    monkeypatch.setattr(
        "qhana_plugin_runner.plugin_utils.dimension_mapping.open_url",
        lambda url, *args, **kwargs: response,
    )

    with pytest.raises(ValueError, match="Could not determine mimetype"):
        load_dimension_mapping(MAPPING_URL)


def test_entity_dimension_names_of_no_entities():
    assert entity_dimension_names([]) == []


def test_entity_dimension_names_of_named_tuples():
    entity = namedtuple("entity", ["ID", "href", "dim0", "dim1"])

    assert entity_dimension_names([entity("e1", "h1", 1, 2)]) == ["dim0", "dim1"]


def test_entity_dimension_names_of_named_tuples_without_href():
    entity = namedtuple("entity", ["ID", "x", "y"])

    assert entity_dimension_names([entity("e1", 1, 2)]) == ["x", "y"]


def test_entity_dimension_names_match_the_ensure_array_value_order():
    """``ensure_array`` sorts dict keys lexicographically, so ``dim10 < dim2``."""
    entity = {"ID": "e1", "href": "h1"}
    entity.update({f"dim{i}": i for i in range(12)})

    names = entity_dimension_names([entity])
    values = next(ensure_array(iter([dict(entity)]))).values

    assert names == [f"dim{i}" for i in sorted(range(12), key=lambda i: f"dim{i}")]
    assert [int(name[len("dim") :]) for name in names] == list(values)


def test_labels_stay_correct_beyond_the_ninth_dimension(monkeypatch):
    """Labels are looked up by name, so the lexicographic value order is harmless."""
    mapping = [
        _mapping_entity(f"dim{i}", f"feature{i}", "dim0", url=f"url{i}")
        for i in range(12)
    ]
    _serve(monkeypatch, mapping)
    labels = load_dimension_mapping(MAPPING_URL)

    entity = {"ID": "e1", "href": "h1"}
    entity.update({f"dim{i}": i for i in range(12)})
    names = entity_dimension_names([entity])
    values = next(ensure_array(iter([dict(entity)]))).values

    assert dimension_labels(names, labels) == [f"feature{value}" for value in values]
    assert dimension_labels(names, labels)[1] == "feature1"


def test_load_dimension_mapping_reads_the_vector_concat_output_verbatim(monkeypatch):
    """The exact payload documented for ``entity/dimension-mapping``."""
    payload = json.loads(
        """
        [
            {"ID": "dim0", "href": "", "inputIndex": 0, "source": "color.json",
             "sourceUrl": "http://localhost:5005/files/17/download/vectors.zip",
             "zipMember": "color.json", "sourceDimension": "dim0"},
            {"ID": "dim1", "href": "", "inputIndex": 0, "source": "color.json",
             "sourceUrl": "http://localhost:5005/files/17/download/vectors.zip",
             "zipMember": "color.json", "sourceDimension": "dim1"},
            {"ID": "dim2", "href": "", "inputIndex": 1, "source": "shape.json",
             "sourceUrl": "http://localhost:5005/files/17/download/vectors.zip",
             "zipMember": "shape.json", "sourceDimension": "dim0"}
        ]
        """
    )
    _serve(monkeypatch, payload)

    assert load_dimension_mapping(MAPPING_URL) == {
        "dim0": "color (dim0)",
        "dim1": "color (dim1)",
        "dim2": "shape",
    }
