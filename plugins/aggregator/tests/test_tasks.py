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
import zipfile

import pytest

from tests.utils import MockResponse, run_plugin_task

from ..tasks import calculation_task
from .data import EXPECTED, TEST_DATA

_MIMETYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "lines": "application/X-lines+json",
}
_ENTITY_FILES = {
    "csv": "entities.csv",
    "json": "entities.json",
    "lines": "entities_lines.json",
}


def _run_aggregator(
    monkeypatch,
    entities_format: str,
    element_distances: dict = None,
):
    entities_file = _ENTITY_FILES[entities_format]
    entities_url = f"http://example.com/{entities_file}"
    metadata_url = "http://example.com/attribute_metadata.json"
    distances_url = "http://example.com/element_distances.zip"

    if element_distances is None:
        element_distances = TEST_DATA["element-distances"]

    responses = {
        entities_url: MockResponse(
            entities_url,
            _MIMETYPES[entities_format],
            text=TEST_DATA[entities_file],
            headers={"X-Attribute-Metadata": metadata_url},
        ),
        metadata_url: MockResponse(
            metadata_url,
            "application/json",
            text=TEST_DATA["attribute_metadata.json"],
        ),
        distances_url: MockResponse.from_zip(distances_url, element_distances),
    }

    return run_plugin_task(
        monkeypatch,
        calculation_task,  # pyright: ignore[reportArgumentType]
        "aggregator.tasks",
        responses,
        {
            "entitiesUrl": entities_url,
            "elementDistancesUrl": distances_url,
        },
    )


def _assert_matches_expected(output):
    assert output.file_type == "relation/attribute-distances"
    assert output.mimetype == "application/zip"

    expected_files = EXPECTED

    with zipfile.ZipFile(output.file_storage_data) as archive:
        assert sorted(info.filename for info in archive.filelist) == sorted(
            expected_files
        )

        for out_filename, expected in expected_files.items():
            actual = json.loads(archive.read(out_filename))

            key = lambda pair: (pair["source"], pair["target"])  # noqa: E731
            actual = sorted(actual, key=key)
            expected = sorted(expected, key=key)

            assert [key(pair) for pair in actual] == [key(pair) for pair in expected]

            for actual_pair, expected_pair in zip(actual, expected):
                if expected_pair["distance"] is None:
                    assert actual_pair["distance"] is None, f"pair {key(actual_pair)}"
                else:
                    assert actual_pair["distance"] == pytest.approx(
                        expected_pair["distance"]
                    ), f"pair {key(actual_pair)}"


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("entities_format", ["csv", "json", "lines"])
def test_aggregator_entity_formats(monkeypatch, entities_format):
    output = _run_aggregator(monkeypatch, entities_format)
    _assert_matches_expected(output)


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("bad_distance", ["null", '"0.5"', "true"])
def test_aggregator_fails_on_non_numeric_element_distance(monkeypatch, bad_distance):
    element_distances = dict(TEST_DATA["element-distances"])
    element_distances["color.json"] = (
        r'[{"source":"red","target":"red","distance":0.0},'
        r'{"source":"red","target":"blue","distance":1.0},'
        r'{"source":"blue","target":"blue","distance":' + bad_distance + r"}]"
    )

    with pytest.raises(ValueError, match="is not a number"):
        _run_aggregator(monkeypatch, "json", element_distances)


@pytest.mark.usefixtures("celery_worker")
def test_aggregator_fails_on_missing_element_pair(monkeypatch):
    element_distances = dict(TEST_DATA["element-distances"])
    # (x, y) is required for the (e1, e2) tags pair but absent
    element_distances["tags.json"] = (
        r'[{"source":"x","target":"x","distance":0.0},'
        r'{"source":"y","target":"y","distance":0.0}]'
    )

    with pytest.raises(ValueError, match="is missing"):
        _run_aggregator(monkeypatch, "json", element_distances)
