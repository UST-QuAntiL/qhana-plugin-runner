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

import pytest

from tests.utils import (
    MockResponse,
    run_plugin_task,
)

from .. import calculation_task
from .data import TEST_DATA

_MIMETYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "lines": "application/X-lines+json",
}
_ENTITY_FILES = {
    "csv": "parts.csv",
    "json": "parts.json",
    "lines": "parts_lines.json",
}
_METADATA_FILES = {
    "csv": "attribute_metadata.csv",
    "json": "attribute_metadata.json",
    "lines": "attribute_metadata_lines.json",
}


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("entities_format", ["csv", "json", "lines"])
@pytest.mark.parametrize("metadata_format", ["csv", "json", "lines"])
def test_one_hot_encoding_matches_expected(monkeypatch, entities_format, metadata_format):
    entities_file = _ENTITY_FILES[entities_format]
    metadata_file = _METADATA_FILES[metadata_format]
    entities_url = f"http://example.com/{entities_file}"
    metadata_url = f"http://example.com/{metadata_file}"
    taxonomies_url = "http://example.com/taxonomies.zip"

    responses = {
        entities_url: MockResponse(
            entities_url,
            _MIMETYPES[entities_format],
            text=TEST_DATA[entities_file],
        ),
        metadata_url: MockResponse(
            metadata_url,
            _MIMETYPES[metadata_format],
            text=TEST_DATA[metadata_file],
        ),
        taxonomies_url: MockResponse.from_zip(
            taxonomies_url,
            {
                "t_AuftretenSatz.json": TEST_DATA["t_AuftretenSatz.json"],
                "t_FormaleFunktion.json": TEST_DATA["t_FormaleFunktion.json"],
                "t_InstrumentierungEinbettungQuantitaet.json": TEST_DATA[
                    "t_InstrumentierungEinbettungQuantitaet.json"
                ],
            },
        ),
    }

    output = run_plugin_task(
        monkeypatch,
        calculation_task,  # pyright: ignore[reportArgumentType]
        "one_hot_encoding",
        responses,
        {
            "entitiesUrl": entities_url,
            "entitiesMetadataUrl": metadata_url,
            "taxonomiesZipUrl": taxonomies_url,
            "attributes": "formal_functions\noccurence_in_movement\ninstrument_quantity_before",
        },
    )

    entities_stem = entities_file.rsplit(".", 1)[0]
    assert output.file_name == f"one-hot-encoded_points_from_{entities_stem}.csv"
    assert output.file_type == "entity/vector"
    assert output.mimetype == "text/csv"

    with open(output.file_storage_data, "r", newline="") as fh:
        result_rows = list(csv.DictReader(fh))
        expected_rows = list(
            csv.DictReader(
                io.StringIO(TEST_DATA["one-hot-encoded_points_from_parts.csv"])
            )
        )
        assert result_rows == expected_rows
