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


def _run_transformer(monkeypatch, transformer: str):
    similaritiesUrl = "http://example.com/element_similarities.zip"

    responses = {
        similaritiesUrl: MockResponse.from_zip(
            similaritiesUrl, TEST_DATA["element-similarities"]
        ),
    }

    return run_plugin_task(
        monkeypatch,
        calculation_task,  # pyright: ignore[reportArgumentType]
        "qhana_plugin_runner.plugin_utils.zip_utils",
        responses,
        {
            "similaritiesUrl": similaritiesUrl,
            "attributes": "color\ntags",
            "transformer": transformer,
        },
    )


def _assert_matches_expected(output, transformer: str):
    # Verify the output format and type as defined in tasks.py
    assert output.file_type == "custom/element-distances"
    assert output.mimetype == "application/zip"

    expected_files = EXPECTED[transformer]

    with zipfile.ZipFile(output.file_storage_data) as archive:
        # Check that exactly the expected json files are inside the zip
        assert sorted(info.filename for info in archive.filelist) == sorted(
            expected_files
        )

        for out_filename, expected in expected_files.items():
            actual = json.loads(archive.read(out_filename))

            key = lambda pair: (pair["source"], pair["target"])  # noqa: E731
            actual = sorted(actual, key=key)
            expected = sorted(expected, key=key)

            # Ensure all entity pairs from the mock data are present in output
            assert [key(pair) for pair in actual] == [key(pair) for pair in expected]

            for actual_pair, expected_pair in zip(actual, expected):
                # Ensure 'similarity' is replaced by 'distance'
                assert "similarity" not in actual_pair
                assert "distance" in actual_pair

                if expected_pair["distance"] is None:
                    assert actual_pair["distance"] is None, f"pair {key(actual_pair)}"
                else:
                    assert actual_pair["distance"] == pytest.approx(
                        expected_pair["distance"]
                    ), f"pair {key(actual_pair)}"


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("transformer", sorted(EXPECTED.keys()))
def test_transformer_scenarios(monkeypatch, transformer):
    """Test all transformer algorithms and verify their calculated outputs."""
    output = _run_transformer(monkeypatch, transformer)
    _assert_matches_expected(output, transformer)
