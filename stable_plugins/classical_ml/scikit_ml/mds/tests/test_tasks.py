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

import json
import math

import pytest

from tests.utils import MockResponse, run_plugin_task

from .. import NONMETRIC_ZERO_FACTOR, _replace_zero_distances, calculation_task
from .data import (
    ENTITY_DISTANCES,
    INCOMPLETE_DISTANCES,
    TINY_DISTANCES,
    ZERO_DISTANCES,
)

_TOLERANCE = 0.05

_PLUGIN_MODULE = _replace_zero_distances.__module__


def _run_mds(
    monkeypatch,
    entity_distances: str = ENTITY_DISTANCES,
    dimensions: int = 2,
    metric: str = "metric_mds",
):
    distances_url = "http://example.com/entity_distances.json"

    responses = {
        distances_url: MockResponse(
            distances_url, "application/json", text=entity_distances
        ),
    }

    return run_plugin_task(
        monkeypatch,
        calculation_task,  # pyright: ignore[reportArgumentType]
        _PLUGIN_MODULE,
        responses,
        {
            "entityDistancesUrl": distances_url,
            "dimensions": dimensions,
            "metric": metric,
            "nInit": 4,
            "maxIter": 300,
        },
    )


def _load_points(output) -> list:
    assert output.file_type == "entity/vector"
    assert output.mimetype == "application/json"
    with open(output.file_storage_data) as file:
        return json.load(file)


def _embedded_distance(points: list, source: str, target: str) -> float:
    by_id = {point["ID"]: point for point in points}
    dims = [key for key in by_id[source] if key.startswith("dim")]
    return math.dist(
        [by_id[source][d] for d in dims],
        [by_id[target][d] for d in dims],
    )


@pytest.mark.usefixtures("celery_worker")
def test_mds_output_format(monkeypatch):
    output = _run_mds(monkeypatch)

    assert (
        output.file_name
        == "entity_points_mds_dim_2_metric_metric_from_entity_distances.json"
    )

    points = _load_points(output)

    assert [point["ID"] for point in points] == ["e1", "e2", "e3"]

    for point in points:
        assert set(point) == {"ID", "href", "dim0", "dim1"}


@pytest.mark.usefixtures("celery_worker")
def test_mds_preserves_distances(monkeypatch):
    output = _run_mds(monkeypatch)
    points = _load_points(output)

    assert _embedded_distance(points, "e1", "e2") == pytest.approx(1.0, abs=_TOLERANCE)
    assert _embedded_distance(points, "e1", "e3") == pytest.approx(2.0, abs=_TOLERANCE)
    assert _embedded_distance(points, "e2", "e3") == pytest.approx(1.0, abs=_TOLERANCE)


def test_replace_zero_distances_only_off_diagonal():
    import numpy as np

    distance_matrix = np.array(
        [
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 2.0],
            [1.5, 2.0, 0.0],
        ]
    )

    _replace_zero_distances(distance_matrix)

    replacement = 1.5 * NONMETRIC_ZERO_FACTOR
    expected = np.array(
        [
            [0.0, replacement, 1.5],
            [replacement, 0.0, 2.0],
            [1.5, 2.0, 0.0],
        ]
    )
    assert np.array_equal(distance_matrix, expected)


def test_replace_zero_distances_below_smallest_positive():
    import numpy as np

    distance_matrix = np.array(
        [
            [0.0, 0.0, 5e-7],
            [0.0, 0.0, 5e-7],
            [5e-7, 5e-7, 0.0],
        ]
    )

    _replace_zero_distances(distance_matrix)

    replacement = distance_matrix[0, 1]
    assert 0 < replacement < 5e-7


def test_replace_zero_distances_underflow_guard():
    import numpy as np

    tiny = 1e-320
    distance_matrix = np.array(
        [
            [0.0, 0.0, tiny],
            [0.0, 0.0, tiny],
            [tiny, tiny, 0.0],
        ]
    )

    _replace_zero_distances(distance_matrix)

    replacement = distance_matrix[0, 1]
    assert 0 < replacement < tiny


def test_replace_zero_distances_all_zero_fallback():
    import numpy as np

    distance_matrix = np.zeros((3, 3))

    _replace_zero_distances(distance_matrix)

    zero_mask = np.eye(3, dtype=bool)
    assert np.all(distance_matrix[~zero_mask] == NONMETRIC_ZERO_FACTOR)
    assert np.all(distance_matrix[zero_mask] == 0)


@pytest.mark.usefixtures("celery_worker")
def test_nonmetric_mds_handles_zero_distances(monkeypatch):
    output = _run_mds(
        monkeypatch, entity_distances=ZERO_DISTANCES, metric="nonmetric_mds"
    )
    points = _load_points(output)

    zero_pair_distance = _embedded_distance(points, "e1", "e2")
    assert zero_pair_distance < _embedded_distance(points, "e1", "e3")
    assert zero_pair_distance < _embedded_distance(points, "e2", "e3")


@pytest.mark.usefixtures("celery_worker")
def test_nonmetric_mds_preserves_order_with_tiny_distances(monkeypatch):
    output = _run_mds(
        monkeypatch, entity_distances=TINY_DISTANCES, metric="nonmetric_mds"
    )
    points = _load_points(output)

    zero_pair_distance = _embedded_distance(points, "e1", "e2")
    assert zero_pair_distance < _embedded_distance(points, "e1", "e3")
    assert zero_pair_distance < _embedded_distance(points, "e2", "e3")


@pytest.mark.usefixtures("celery_worker")
def test_metric_mds_keeps_zero_distances(monkeypatch):
    output = _run_mds(monkeypatch, entity_distances=ZERO_DISTANCES)
    points = _load_points(output)

    assert _embedded_distance(points, "e1", "e2") == pytest.approx(0.0, abs=_TOLERANCE)
    assert _embedded_distance(points, "e1", "e3") == pytest.approx(1.0, abs=_TOLERANCE)
    assert _embedded_distance(points, "e2", "e3") == pytest.approx(1.0, abs=_TOLERANCE)


@pytest.mark.usefixtures("celery_worker")
def test_mds_fails_when_pair_has_no_distance_entry(monkeypatch):
    with pytest.raises(
        ValueError,
        match=r"no distance entry for entity pairs: \(e2, e3\)",
    ):
        _run_mds(monkeypatch, entity_distances=INCOMPLETE_DISTANCES)
