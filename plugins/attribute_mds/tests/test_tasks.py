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
import zipfile
from pathlib import PurePath

import pytest

from qhana_plugin_runner.plugin_utils.hashing import get_readable_hash
from tests.utils import MockResponse, run_plugin_task

from ..tasks import calculation_task
from .data import ALL_MISSING_DISTANCES, ATTRIBUTE_DISTANCES

_TOLERANCE = 0.05


def _run_mds(
    monkeypatch,
    attribute_distances: dict = None,
    missing_data_handling: str = "mean",
    dimensions: int = 2,
):
    distances_url = "http://example.com/attribute_distances.zip"

    if attribute_distances is None:
        attribute_distances = ATTRIBUTE_DISTANCES

    responses = {
        distances_url: MockResponse.from_zip(distances_url, attribute_distances),
    }

    return run_plugin_task(
        monkeypatch,
        calculation_task,  # pyright: ignore[reportArgumentType]
        "qhana_plugin_runner.requests",
        responses,
        {
            "attributeDistancesUrl": distances_url,
            "dimensions": dimensions,
            "metric": "metric_mds",
            "nInit": 4,
            "maxIter": 300,
            "missingDataHandling": missing_data_handling,
        },
    )


def _points_by_attribute(output) -> dict:
    assert output.file_type == "entity/vector"
    assert output.mimetype == "application/zip"

    points = {}
    with zipfile.ZipFile(output.file_storage_data) as archive:
        for member in archive.namelist():
            attr_name = PurePath(member).stem
            points[attr_name] = json.loads(archive.read(member))
    return points


def _embedded_distance(points: list, source: str, target: str) -> float:
    by_id = {point["ID"]: point for point in points}
    dims = [key for key in by_id[source] if key.startswith("dim")]
    return math.dist(
        [by_id[source][d] for d in dims],
        [by_id[target][d] for d in dims],
    )


@pytest.mark.usefixtures("celery_worker")
def test_mds_output_per_attribute(monkeypatch):
    output = _run_mds(monkeypatch)

    filenames_hash = get_readable_hash("attribute_distances")
    assert (
        output.file_name
        == f"entity_points_mds_dim_2_metric_mean_from_attribute_distances_{filenames_hash}.zip"
    )

    points = _points_by_attribute(output)

    assert sorted(points) == ["color", "size"]

    id_orders = [[point["ID"] for point in pts] for pts in points.values()]
    assert id_orders[0] == ["e1", "e2", "e3"]
    assert id_orders[0] == id_orders[1]

    for pts in points.values():
        for point in pts:
            assert set(point) == {"ID", "href", "dim0", "dim1"}


@pytest.mark.usefixtures("celery_worker")
def test_mds_preserves_distances(monkeypatch):
    output = _run_mds(monkeypatch)
    points = _points_by_attribute(output)

    color = points["color"]
    assert _embedded_distance(color, "e1", "e2") == pytest.approx(1.0, abs=_TOLERANCE)
    assert _embedded_distance(color, "e1", "e3") == pytest.approx(2.0, abs=_TOLERANCE)
    assert _embedded_distance(color, "e2", "e3") == pytest.approx(1.0, abs=_TOLERANCE)


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize(
    ("missing_data_handling", "expected_replacement"),
    [("mean", 0.7), ("max", 0.8)],
)
def test_mds_missing_distance_replacement(
    monkeypatch, missing_data_handling, expected_replacement
):
    output = _run_mds(monkeypatch, missing_data_handling=missing_data_handling)
    points = _points_by_attribute(output)

    size = points["size"]
    assert _embedded_distance(size, "e1", "e2") == pytest.approx(0.6, abs=_TOLERANCE)
    assert _embedded_distance(size, "e1", "e3") == pytest.approx(0.8, abs=_TOLERANCE)
    assert _embedded_distance(size, "e2", "e3") == pytest.approx(
        expected_replacement, abs=_TOLERANCE
    )


@pytest.mark.usefixtures("celery_worker")
def test_mds_fails_when_all_distances_missing(monkeypatch):
    with pytest.raises(ValueError, match="every distance for attribute 'color'"):
        _run_mds(monkeypatch, attribute_distances=ALL_MISSING_DISTANCES)
