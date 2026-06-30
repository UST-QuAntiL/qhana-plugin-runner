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

"""Unit tests for the pure helper functions in ``mapping_distances.tasks``.

These cover the building blocks of the distance calculation in isolation:

* :func:`~mapping_distances.tasks.extract_tax_name`
* :func:`~mapping_distances.tasks.get_element_list`
* :func:`~mapping_distances.tasks.calculate_vector_distance`
* :func:`~mapping_distances.tasks.load_input_parameters`

The first three are plain functions with no Flask/DB/Celery dependency.
``load_input_parameters`` reads from the database and therefore uses the
shared ``task_data`` fixture from the repo-root ``conftest.py``.
"""

import math
import sys

import pytest

from mapping_distances.schemas import (
    DistanceMetricEnum,
    InputParameters,
    InputParametersSchema,
)
from mapping_distances.tasks import (
    calculate_vector_distance,
    extract_tax_name,
    get_element_list,
    load_input_parameters,
)
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata


def _meta(**overrides) -> AttributeMetadata:
    """Build an ``AttributeMetadata`` with sensible defaults for tests."""
    defaults = dict(ID="color", attribute_type="ref", title="Color")
    defaults.update(overrides)
    return AttributeMetadata(**defaults)


# ---------------------------------------------------------------------------
# extract_tax_name
# ---------------------------------------------------------------------------


class TestExtractTaxName:
    def test_ref_target_with_json_suffix(self):
        meta = _meta(ref_target="taxonomies.zip:color.json")
        assert extract_tax_name(meta) == "color"

    def test_ref_target_without_json_suffix(self):
        meta = _meta(ref_target="taxonomies.zip:color")
        assert extract_tax_name(meta) == "color"

    def test_ref_target_nested_name(self):
        meta = _meta(ref_target="taxonomies.zip:some/nested.json")
        assert extract_tax_name(meta) == "some/nested"

    def test_ref_target_none(self):
        assert extract_tax_name(_meta(ref_target=None)) == ""

    def test_ref_target_without_marker(self):
        meta = _meta(ref_target="something_else.json")
        assert extract_tax_name(meta) == ""

    def test_metadata_none(self):
        assert extract_tax_name(None) == ""


# ---------------------------------------------------------------------------
# get_element_list
# ---------------------------------------------------------------------------


class TestGetElementList:
    def test_missing_attribute_returns_empty(self):
        assert get_element_list({"ID": "e1"}, "color", _meta()) == []

    def test_none_value_returns_empty(self):
        assert get_element_list({"color": None}, "color", _meta()) == []

    @pytest.mark.parametrize("empty", [[], set(), {}])
    def test_empty_collection_returns_empty(self, empty):
        assert get_element_list({"color": empty}, "color", _meta()) == []

    def test_list_value_is_stringified_and_filtered(self):
        result = get_element_list({"color": ["red", "", "blue"]}, "color", _meta())
        assert result == ["red", "blue"]

    def test_set_value_is_stringified(self):
        result = get_element_list({"color": {"red"}}, "color", _meta())
        assert result == ["red"]

    def test_single_string_value(self):
        result = get_element_list({"color": "red"}, "color", _meta())
        assert result == ["red"]

    def test_blank_string_value_returns_empty(self):
        assert get_element_list({"color": "   "}, "color", _meta()) == []

    def test_multiple_string_value_is_split_on_separator(self):
        meta = _meta(multiple=True, separator=";")
        result = get_element_list({"color": "red; blue ;green"}, "color", meta)
        assert result == ["red", "blue", "green"]

    def test_multiple_without_separator_is_treated_as_single(self):
        meta = _meta(multiple=True, separator="")
        result = get_element_list({"color": "red;blue"}, "color", meta)
        assert result == ["red;blue"]

    def test_non_string_scalar_is_stringified(self):
        assert get_element_list({"color": 42}, "color", _meta()) == ["42"]


# ---------------------------------------------------------------------------
# calculate_vector_distance
# ---------------------------------------------------------------------------


class TestCalculateVectorDistance:
    def test_euclidean(self):
        dist = calculate_vector_distance(
            [0.0, 0.0], [3.0, 4.0], DistanceMetricEnum.euclidean, 0
        )
        assert dist == pytest.approx(5.0)

    def test_manhatten(self):
        dist = calculate_vector_distance(
            [0.0, 0.0], [3.0, 4.0], DistanceMetricEnum.manhatten, 0
        )
        assert dist == pytest.approx(7.0)

    def test_chebyshev(self):
        dist = calculate_vector_distance(
            [0.0, 0.0], [3.0, 4.0], DistanceMetricEnum.chebyshev, 0
        )
        assert dist == pytest.approx(4.0)

    def test_cosine_orthogonal_vectors(self):
        dist = calculate_vector_distance(
            [1.0, 0.0], [0.0, 1.0], DistanceMetricEnum.cosine, 0
        )
        assert dist == pytest.approx(1.0)

    def test_cosine_identical_direction(self):
        dist = calculate_vector_distance(
            [1.0, 2.0], [2.0, 4.0], DistanceMetricEnum.cosine, 0
        )
        assert dist == pytest.approx(0.0, abs=1e-9)

    def test_cosine_with_zero_vector_returns_max_cosine_distance(self):
        dist = calculate_vector_distance(
            [0.0, 0.0], [1.0, 1.0], DistanceMetricEnum.cosine, 0
        )
        assert dist == 2

    def test_empty_vectors_return_float_max(self):
        dist = calculate_vector_distance([], [], DistanceMetricEnum.euclidean, 0)
        assert dist == sys.float_info.max

    def test_mismatched_lengths_raise_value_error(self):
        with pytest.raises(ValueError, match="same length"):
            calculate_vector_distance([1.0], [1.0, 2.0], DistanceMetricEnum.euclidean, 0)

    def test_unknown_metric_raises_value_error(self):
        class _FakeMetric:
            name = "fake"

        with pytest.raises(ValueError, match="Unknown distance metric"):
            calculate_vector_distance([1.0], [2.0], _FakeMetric(), 0)

    def test_euclidean_result_is_finite(self):
        dist = calculate_vector_distance(
            [1.5, -2.0], [0.5, 3.0], DistanceMetricEnum.euclidean, 0
        )
        assert math.isfinite(dist)


# ---------------------------------------------------------------------------
# load_input_parameters (needs a Flask app context + DB row)
# ---------------------------------------------------------------------------


class TestLoadInputParameters:
    def test_parses_parameters_and_splits_attributes(self, task_data):
        params = InputParameters(
            entities_url="file:///entities.json",
            entities_metadata_url="file:///metadata.json",
            taxonomies_zip_url="file:///taxonomies.zip",
            attributes="color\n\n  size  \n",
            distance_metric=DistanceMetricEnum.manhatten,
        )
        task_data.parameters = InputParametersSchema().dumps(params)
        task_data.save(commit=True)

        (
            entities_url,
            entities_metadata_url,
            taxonomies_zip_url,
            attributes,
            distance_metric,
        ) = load_input_parameters(task_data.id)

        assert entities_url == "file:///entities.json"
        assert entities_metadata_url == "file:///metadata.json"
        assert taxonomies_zip_url == "file:///taxonomies.zip"
        # Blank lines are dropped and surrounding whitespace is stripped.
        assert attributes == ["color", "size"]
        assert distance_metric == DistanceMetricEnum.manhatten

    def test_missing_db_id_raises_key_error(self, task_data):
        with pytest.raises(KeyError, match="Could not load task data"):
            load_input_parameters(999999)
