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


import pytest
from marshmallow import ValidationError

from vector_concat.schemas import VectorConcatSchema

VALID_URL = "http://example.com/data.csv"
SECOND_URL = "https://example.com/other.json"


def _payload(urls: str, **overrides) -> dict:
    base = {"urls": urls}
    base.update(overrides)
    return base


def test_single_url_valid_defaults_output_format_to_csv():
    result = VectorConcatSchema().load(_payload(VALID_URL))
    assert isinstance(result, dict)
    assert result["urls"] == VALID_URL
    assert result["output_format"] == "csv"


def test_multiple_urls_valid():
    urls = f"{VALID_URL}\n{SECOND_URL}"
    result = VectorConcatSchema().load(_payload(urls))
    assert isinstance(result, dict)
    assert result["urls"] == urls
    assert result["output_format"] == "csv"


@pytest.mark.parametrize("fmt", ["csv", "json", "lines"])
def test_output_format_accepted(fmt):
    result = VectorConcatSchema().load(_payload(VALID_URL, outputFormat=fmt))
    assert isinstance(result, dict)
    assert result["output_format"] == fmt


def test_blank_lines_between_valid_urls_tolerated():
    urls = f"{VALID_URL}\n\n   \n{SECOND_URL}"
    result = VectorConcatSchema().load(_payload(urls))
    assert isinstance(result, dict)
    assert result["urls"] == urls


def test_unknown_output_format_rejected():
    with pytest.raises(ValidationError) as exc:
        VectorConcatSchema().load(_payload(VALID_URL, outputFormat="xml"))
    assert exc.value.messages == {"outputFormat": ["Must be one of: csv, json, lines."]}


def test_urls_required():
    with pytest.raises(ValidationError) as exc:
        VectorConcatSchema().load({})
    assert exc.value.messages == {"urls": ["Missing data for required field."]}


@pytest.mark.parametrize("value", ["", "   ", "\n\n", "  \n \n"])
def test_empty_or_whitespace_urls_rejected(value):
    with pytest.raises(ValidationError) as exc:
        VectorConcatSchema().load(_payload(value))
    assert exc.value.messages == {"urls": ["At least one URL is required."]}


def test_single_invalid_url_reports_line_one():
    with pytest.raises(ValidationError) as exc:
        VectorConcatSchema().load(_payload("not-a-url"))
    assert exc.value.messages == {"urls": ["Line 1: Not a valid URL."]}


def test_mixed_valid_and_invalid_reports_only_offending_line():
    urls = f"{VALID_URL}\nnot-a-url"
    with pytest.raises(ValidationError) as exc:
        VectorConcatSchema().load(_payload(urls))
    assert exc.value.messages == {"urls": ["Line 2: Not a valid URL."]}
