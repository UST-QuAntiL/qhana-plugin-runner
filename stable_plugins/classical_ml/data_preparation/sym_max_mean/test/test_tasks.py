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

from tests.utils import (
    MockResponse,
    run_plugin_task,
)

from .. import calculation_task
from .data import TEST_DATA

# _INPUTS = TEST_DATA["input"]
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


def _normalize(similarities):
    return sorted(similarities, key=lambda pair: json.dumps(pair, sort_keys=True))


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("entities_format", ["csv", "json", "lines"])
@pytest.mark.parametrize("metadata_format", ["csv", "json", "lines"])
def test_sym_max_mean_matches_expected(monkeypatch, entities_format, metadata_format):
    entities_file = _ENTITY_FILES[entities_format]
    metadata_file = _METADATA_FILES[metadata_format]
    entities_url = f"http://example.com/{entities_file}"
    metadata_url = f"http://example.com/{metadata_file}"
    similarities_url = "http://example.com/element_similarities.zip"

    responses = {
        entities_url: MockResponse(
            entities_url,
            _MIMETYPES[entities_format],
            text=TEST_DATA[entities_file],
            headers={"X-Attribute-Metadata": metadata_url},
        ),
        metadata_url: MockResponse(
            metadata_url,
            _MIMETYPES[metadata_format],
            text=TEST_DATA[metadata_file],
        ),
        similarities_url: MockResponse.from_zip(
            similarities_url,
            {
                "formal_functions.json": TEST_DATA["element-similarities"][
                    "formal_functions.json"
                ],
                "instrument_quantity_before.json": TEST_DATA["element-similarities"][
                    "instrument_quantity_before.json"
                ],
                "occurence_in_movement.json": TEST_DATA["element-similarities"][
                    "occurence_in_movement.json"
                ],
            },
        ),
    }

    output = run_plugin_task(
        monkeypatch,
        calculation_task,  # pyright: ignore[reportArgumentType]
        "sym_max_mean",  # TODO replace with __name__ (or similar)
        responses,
        {
            "entities_url": entities_url,
            "element_similarities_url": similarities_url,
            "attributes": "formal_functions\noccurence_in_movement\ninstrument_quantity_before",
        },
    )

    assert output.file_type == "relation/attribute-similarities"
    assert output.mimetype == "application/zip"

    with zipfile.ZipFile(output.file_storage_data) as archive:
        assert len(archive.filelist) == 3

        for out_filename in (
            "formal_functions.json",
            "occurence_in_movement.json",
            "instrument_quantity_before.json",
        ):
            output_file = _normalize(json.loads(archive.read(out_filename)))
            expected = _normalize(
                json.loads(TEST_DATA["attribute-similarities"][out_filename])
            )
            assert output_file == expected
