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

import csv
import io
import json

import pytest

from tests.utils import MockResponse, run_plugin_task

from .. import convert_csv, convert_json
from .data import (
    EXPECTED_CSV_ROWS,
    EXPECTED_ENTITIES,
    EXPECTED_VECTOR_ENTITIES,
    TEST_DATA,
)

_MIMETYPES = {
    "csv": "text/csv",
    "json": "application/json",
}


def _run_conversion(
    monkeypatch,
    task,
    entities_file: str,
    mimetype: str,
    *,
    data_type: str = "entity/list",
    with_attribute_metadata: bool = True,
    entities_text: str = None,
):
    entities_url = f"http://example.com/{entities_file}"
    metadata_url = "http://example.com/attribute_metadata.json"

    entities_headers = {"X-Data-Type": data_type}
    if with_attribute_metadata:
        entities_headers["X-Attribute-Metadata"] = metadata_url

    if entities_text is None:
        entities_text = TEST_DATA[entities_file]

    responses = {
        entities_url: MockResponse(
            entities_url,
            mimetype,
            text=entities_text,
            headers=entities_headers,
        ),
        metadata_url: MockResponse(
            metadata_url,
            "application/json",
            text=TEST_DATA["attribute_metadata.json"],
        ),
    }

    return run_plugin_task(
        monkeypatch,
        task,
        "entity_conversion",
        responses,
        entities_url,
    )


def _read_output_text(output) -> str:
    # ``file_storage_data`` is the path of the file in the local file store.
    with open(output.file_storage_data, encoding="utf-8") as file:
        return file.read()


def _read_csv_rows(output) -> list:
    return list(csv.reader(io.StringIO(_read_output_text(output))))


@pytest.mark.usefixtures("celery_worker")
def test_csv_to_json_matches_expected(monkeypatch):
    output = _run_conversion(monkeypatch, convert_csv, "entities.csv", "text/csv")

    assert output.file_type == "entity/list"
    assert output.mimetype == "application/json"
    assert json.loads(_read_output_text(output)) == EXPECTED_ENTITIES


@pytest.mark.usefixtures("celery_worker")
def test_json_to_csv_matches_expected(monkeypatch):
    output = _run_conversion(
        monkeypatch, convert_json, "entities.json", "application/json"
    )

    assert output.file_type == "entity/list"
    assert output.mimetype == "text/csv"
    assert _read_csv_rows(output) == EXPECTED_CSV_ROWS


@pytest.mark.usefixtures("celery_worker")
def test_conversion_round_trip_preserves_entities(monkeypatch):
    json_output = _run_conversion(monkeypatch, convert_csv, "entities.csv", "text/csv")
    json_text = _read_output_text(json_output)

    csv_output = _run_conversion(
        monkeypatch,
        convert_json,
        "entities.json",
        "application/json",
        entities_text=json_text,
    )

    assert _read_csv_rows(csv_output) == EXPECTED_CSV_ROWS


@pytest.mark.usefixtures("celery_worker")
def test_csv_to_json_vector_without_metadata(monkeypatch):
    output = _run_conversion(
        monkeypatch,
        convert_csv,
        "vector.csv",
        "text/csv",
        data_type="entity/vector",
        with_attribute_metadata=False,
    )

    assert output.file_type == "entity/vector"
    assert output.mimetype == "application/json"
    assert json.loads(_read_output_text(output)) == EXPECTED_VECTOR_ENTITIES


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize(
    "entities_format,entities_text",
    [("csv", '"ID","href"'), ("json", "[]")],
)
def test_conversion_fails_on_empty_input(monkeypatch, entities_format, entities_text):
    task = convert_csv if entities_format == "csv" else convert_json
    entities_file = f"entities.{entities_format}"

    with pytest.raises(ValueError, match="Cannot convert an empty list of entities"):
        _run_conversion(
            monkeypatch,
            task,
            entities_file,
            _MIMETYPES[entities_format],
            entities_text=entities_text,
        )


@pytest.mark.usefixtures("celery_worker")
def test_conversion_fails_on_unsupported_mimetype(monkeypatch):
    lines_text = "\n".join(
        json.dumps(entity) for entity in json.loads(TEST_DATA["entities.json"])
    )

    with pytest.raises(ValueError, match="Unsupported mimetype"):
        _run_conversion(
            monkeypatch,
            convert_json,
            "entities_lines.json",
            "application/X-lines+json",
            entities_text=lines_text,
        )
