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

import pytest

from qhana_plugin_runner.plugin_utils.dimension_mapping import load_dimension_labels

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


def _labels_of(monkeypatch, mapping):
    _serve(monkeypatch, mapping)
    return load_dimension_labels(MAPPING_URL)


def test_every_dimension_of_a_source_shares_the_feature_name(monkeypatch):
    labels = _labels_of(
        monkeypatch,
        [
            _mapping_entity("dim0", "color", "dim0", url="color-url"),
            _mapping_entity("dim1", "color", "dim1", url="color-url"),
            _mapping_entity("dim2", "shape", "dim0", url="shape-url"),
        ],
    )

    assert labels == {
        "dim0": "color",
        "dim1": "color",
        "dim2": "shape",
    }


def test_label_ignores_the_source_column_name(monkeypatch):
    labels = _labels_of(
        monkeypatch,
        [
            _mapping_entity("dim0", "points", "x"),
            _mapping_entity("dim1", "points", "y"),
        ],
    )

    assert labels == {"dim0": "points", "dim1": "points"}


def test_zip_members_and_plain_urls_yield_the_same_name(monkeypatch):
    from_zip = _labels_of(
        monkeypatch,
        [_mapping_entity("dim0", "color.json", "dim0", zip_member="color.json")],
    )
    from_url = _labels_of(monkeypatch, [_mapping_entity("dim0", "color", "dim0")])

    assert from_zip == from_url == {"dim0": "color"}


@pytest.mark.parametrize("source", [None, "", "  "])
def test_entities_without_a_source_are_skipped(monkeypatch, source):
    labels = _labels_of(
        monkeypatch,
        [
            _mapping_entity("dim0", source, "dim0"),
            _mapping_entity("dim1", "color", "dim0"),
        ],
    )

    assert labels == {"dim1": "color"}


def test_entities_without_a_dimension_name_are_skipped(monkeypatch):
    assert _labels_of(monkeypatch, [_mapping_entity("", "color", "dim0")]) == {}


def test_a_missing_source_dimension_does_not_change_the_label(monkeypatch):
    labels = _labels_of(
        monkeypatch,
        [
            _mapping_entity("dim0", "color", ""),
            _mapping_entity("dim1", "color", "dim1"),
        ],
    )

    assert labels == {"dim0": "color", "dim1": "color"}


@pytest.mark.parametrize("url", [None, ""])
def test_no_url_yields_no_labels(url):
    assert load_dimension_labels(url) == {}


def test_load_dimension_labels_reads_the_file(monkeypatch):
    _serve(
        monkeypatch,
        [
            _mapping_entity("dim0", "color", "dim0", url="color-url"),
            _mapping_entity("dim1", "color", "dim1", url="color-url"),
            _mapping_entity("dim2", "shape", "dim0", url="shape-url"),
        ],
    )

    assert load_dimension_labels(MAPPING_URL) == {
        "dim0": "color",
        "dim1": "color",
        "dim2": "shape",
    }


def test_load_dimension_labels_without_mimetype_raises(monkeypatch):
    # A url without a file extension gives no mimetype hint.
    response = MockResponse("http://example.com/download", "", json_data=[])
    del response.headers["Content-Type"]
    monkeypatch.setattr(
        "qhana_plugin_runner.plugin_utils.dimension_mapping.open_url",
        lambda url, *args, **kwargs: response,
    )

    with pytest.raises(ValueError, match="Could not determine mimetype"):
        load_dimension_labels(MAPPING_URL)


def test_load_dimension_labels_reads_the_vector_concat_output_verbatim(monkeypatch):
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

    assert load_dimension_labels(MAPPING_URL) == {
        "dim0": "color",
        "dim1": "color",
        "dim2": "shape",
    }
