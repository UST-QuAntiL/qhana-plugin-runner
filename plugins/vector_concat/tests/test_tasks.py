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
import json

import pytest

from tests.utils import MockResponse, run_plugin_task_outputs

from vector_concat.tasks import calculation_task

from .data import EXTENSIONS, MIMETYPES, TEST_DATA


def _input_filename(base: str, fmt: str) -> str:
    """Filename of the ``base`` source serialized as ``fmt`` (matches data.py)."""
    return f"{base}_lines.json" if fmt == "lines" else f"{base}.{fmt}"


def _response(base: str, fmt: str) -> tuple[str, MockResponse]:
    filename = _input_filename(base, fmt)
    url = f"http://example.com/{filename}"
    return url, MockResponse(url, MIMETYPES[fmt], text=TEST_DATA[filename])


def _read_csv(output):
    with open(output.file_storage_data, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(output):
    with open(output.file_storage_data, "r") as fh:
        return json.load(fh)


def _read_lines(output):
    with open(output.file_storage_data, "r") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_READERS = {"csv": _read_csv, "json": _read_json, "lines": _read_lines}


def _run_concat(monkeypatch, responses, params):
    """Run the task and return its vector and dimension mapping outputs."""
    outputs = run_plugin_task_outputs(
        monkeypatch,
        calculation_task,  # pyright: ignore[reportArgumentType]
        "vector_concat.tasks",
        responses,
        params,
    )

    by_type = {output.file_type: output for output in outputs}
    assert len(outputs) == 2
    assert set(by_type) == {"entity/vector", "entity/dimension-mapping"}

    return by_type["entity/vector"], by_type["entity/dimension-mapping"]


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("output_format", ["csv", "json", "lines"])
def test_concat_two_inputs_per_output_format(monkeypatch, output_format):
    url_a, response_a = _response("a", output_format)
    url_b, response_b = _response("b", output_format)
    responses = {url_a: response_a, url_b: response_b}

    output, _mapping = _run_concat(
        monkeypatch,
        responses,
        {"urls": f"{url_a}\n{url_b}", "output_format": output_format},
    )

    assert output.file_name == f"concatenated.{EXTENSIONS[output_format]}"
    assert output.file_type == "entity/vector"
    assert output.mimetype == MIMETYPES[output_format]

    rows = _READERS[output_format](output)
    assert len(rows) == 1
    row = rows[0]
    assert row["ID"] == "e1"
    assert row["href"] == "h1"
    # csv reads values back as strings, json and lines as ints.
    assert [int(row[f"dim{i}"]) for i in range(4)] == [1, 2, 3, 4]


@pytest.mark.usefixtures("celery_worker")
def test_concat_mixed_input_formats(monkeypatch):
    """Inputs of different content types combine into one output."""
    url_a, response_a = _response("a", "csv")
    url_b, response_b = _response("b", "lines")
    responses = {url_a: response_a, url_b: response_b}

    output, _mapping = _run_concat(
        monkeypatch,
        responses,
        {"urls": f"{url_a}\n{url_b}", "output_format": "json"},
    )

    entities = _read_json(output)
    assert [entities[0][f"dim{i}"] for i in range(4)] == [1, 2, 3, 4]


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("member_format", ["csv", "json", "lines"])
def test_concat_zip_input(monkeypatch, member_format):
    """A zip of entity/vector files is unpacked and each member is concatenated."""
    url_a, response_a = _response("a", "csv")
    zip_url = "http://example.com/vectors.zip"
    zip_member = _input_filename("b", member_format)
    zip_response = MockResponse.from_zip(zip_url, {zip_member: TEST_DATA[zip_member]})
    responses = {url_a: response_a, zip_url: zip_response}

    output, _mapping = _run_concat(
        monkeypatch,
        responses,
        {"urls": f"{url_a}\n{zip_url}", "output_format": "json"},
    )

    entities = _read_json(output)
    assert len(entities) == 1
    assert entities[0]["ID"] == "e1"
    assert entities[0]["href"] == "h1"
    assert [entities[0][f"dim{i}"] for i in range(4)] == [1, 2, 3, 4]


@pytest.mark.usefixtures("celery_worker")
def test_concat_zip_multiple_members(monkeypatch):
    """Every file inside a zip contributes its own dimensions to the output."""
    zip_url = "http://example.com/vectors.zip"
    zip_response = MockResponse.from_zip(
        zip_url,
        {
            "a.csv": TEST_DATA["a.csv"],
            "b.json": TEST_DATA["b.json"],
        },
    )

    output, _mapping = _run_concat(
        monkeypatch,
        {zip_url: zip_response},
        {"urls": zip_url, "output_format": "json"},
    )

    entities = _read_json(output)
    assert len(entities) == 1
    # members are processed in sorted name order: a.csv then b.json
    assert [entities[0][f"dim{i}"] for i in range(4)] == [1, 2, 3, 4]


@pytest.mark.usefixtures("celery_worker")
def test_concat_zip_without_member_extensions_raises(monkeypatch):
    """Zip members without an extension do not provide a mimetype."""
    zip_url = "http://example.com/vectors.zip"
    zip_response = MockResponse.from_zip(
        zip_url,
        {
            "first": TEST_DATA["a.csv"],  # No file extension
            "second": TEST_DATA["b.json"],  # No file extension
        },
    )

    with pytest.raises(ValueError, match=r"No Mimetype found for zip file 'first'\."):
        _run_concat(
            monkeypatch,
            {zip_url: zip_response},
            {"urls": zip_url, "output_format": "json"},
        )


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("output_format", ["csv", "json", "lines"])
def test_output_suffix_appends_to_filename(monkeypatch, output_format):
    url_a, response_a = _response("a", output_format)
    url_b, response_b = _response("b", output_format)
    responses = {url_a: response_a, url_b: response_b}

    output, _mapping = _run_concat(
        monkeypatch,
        responses,
        {
            "urls": f"{url_a}\n{url_b}",
            "output_format": output_format,
            "output_suffix": "run42",
        },
    )

    assert output.file_name == f"concatenated_run42.{EXTENSIONS[output_format]}"


@pytest.mark.usefixtures("celery_worker")
def test_empty_output_suffix_keeps_default_filename(monkeypatch):
    url, response = _response("single", "csv")

    output, _mapping = _run_concat(
        monkeypatch,
        {url: response},
        {"urls": url, "output_suffix": "   "},
    )

    assert output.file_name == "concatenated.csv"


@pytest.mark.usefixtures("celery_worker")
def test_output_format_defaults_to_csv(monkeypatch):
    """When ``output_format`` is absent the task falls back to CSV output."""
    url, response = _response("single", "csv")

    output, _mapping = _run_concat(
        monkeypatch,
        {url: response},
        {"urls": url},
    )

    assert output.file_name == "concatenated.csv"
    assert output.mimetype == "text/csv"


@pytest.mark.usefixtures("celery_worker")
def test_mismatched_ids_raise(monkeypatch):
    url_a, response_a = _response("a", "csv")
    url_b = "http://example.com/mismatch.csv"
    mismatch = "ID,href,dim0,dim1\ne2,h1,3,4"
    responses = {
        url_a: response_a,
        url_b: MockResponse(url_b, "text/csv", text=mismatch),
    }

    with pytest.raises(AssertionError):
        _run_concat(
            monkeypatch,
            responses,
            {"urls": f"{url_a}\n{url_b}", "output_format": "csv"},
        )


@pytest.mark.usefixtures("celery_worker")
def test_unequal_row_counts_raise(monkeypatch):
    """``zip(*entities_list, strict=True)`` rejects inputs of differing length."""
    url_a = "http://example.com/two_rows.csv"
    url_b, response_b = _response("b", "csv")
    two_rows = "ID,href,dim0,dim1\ne1,h1,1,2\ne2,h2,5,6"
    responses = {
        url_a: MockResponse(url_a, "text/csv", text=two_rows),
        url_b: response_b,
    }

    with pytest.raises(ValueError):
        _run_concat(
            monkeypatch,
            responses,
            {"urls": f"{url_a}\n{url_b}", "output_format": "csv"},
        )


@pytest.mark.usefixtures("celery_worker")
def test_ragged_entities_raise(monkeypatch):
    """Entities of differing width cannot be mapped to a fixed dimension set."""
    url = "http://example.com/ragged.json"
    responses = {
        url: MockResponse(url, "application/json", text=TEST_DATA["ragged.json"])
    }

    with pytest.raises(ValueError, match=r"Entity 'e2' has 1 dimensions"):
        _run_concat(monkeypatch, responses, {"urls": url, "output_format": "json"})


@pytest.mark.usefixtures("celery_worker")
def test_missing_db_id_raises(monkeypatch):
    """Task raises ``KeyError`` when no ``ProcessingTask`` row matches the id."""
    with pytest.raises(KeyError, match="Could not load task data"):
        calculation_task.apply_async(kwargs={"db_id": 99999}).get(timeout=30)


@pytest.mark.usefixtures("celery_worker")
@pytest.mark.parametrize("output_format", ["csv", "json", "lines"])
def test_dimension_mapping_for_plain_inputs(monkeypatch, output_format):
    """Every output dimension is traced back to its input file and column."""
    url_a, response_a = _response("a", output_format)
    url_b, response_b = _response("b", output_format)
    responses = {url_a: response_a, url_b: response_b}

    _output, mapping_output = _run_concat(
        monkeypatch,
        responses,
        {"urls": f"{url_a}\n{url_b}", "output_format": output_format},
    )

    assert mapping_output.file_name == "concatenated_dimension_mapping.json"
    assert mapping_output.file_type == "entity/dimension-mapping"
    # the mapping is always JSON, independent of the output format
    assert mapping_output.mimetype == "application/json"

    # the source name is the file name of the input url without its extension
    name_a = _input_filename("a", output_format).split(".")[0]
    name_b = _input_filename("b", output_format).split(".")[0]

    mapping = _read_json(mapping_output)
    assert [
        (row["ID"], row["inputIndex"], row["source"], row["sourceDimension"])
        for row in mapping
    ] == [
        ("dim0", 0, name_a, "dim0"),
        ("dim1", 0, name_a, "dim1"),
        ("dim2", 1, name_b, "dim0"),
        ("dim3", 1, name_b, "dim1"),
    ]
    assert [row["sourceUrl"] for row in mapping] == [url_a, url_a, url_b, url_b]
    assert all(row["zipMember"] == "" for row in mapping)


@pytest.mark.usefixtures("celery_worker")
def test_dimension_mapping_records_zip_members(monkeypatch):
    """Zip members are identified by their name inside the archive."""
    zip_url = "http://example.com/vectors.zip"
    zip_response = MockResponse.from_zip(
        zip_url,
        {
            "b.json": TEST_DATA["b.json"],
            "a.csv": TEST_DATA["a.csv"],
        },
    )

    _output, mapping_output = _run_concat(
        monkeypatch,
        {zip_url: zip_response},
        {"urls": zip_url, "output_format": "json"},
    )

    mapping = _read_json(mapping_output)
    # members are read in sorted name order, not in archive order
    assert [
        (row["ID"], row["inputIndex"], row["source"], row["sourceDimension"])
        for row in mapping
    ] == [
        ("dim0", 0, "a.csv", "dim0"),
        ("dim1", 0, "a.csv", "dim1"),
        ("dim2", 1, "b.json", "dim0"),
        ("dim3", 1, "b.json", "dim1"),
    ]
    assert [row["zipMember"] for row in mapping] == [
        "a.csv",
        "a.csv",
        "b.json",
        "b.json",
    ]
    assert all(row["sourceUrl"] == zip_url for row in mapping)


@pytest.mark.usefixtures("celery_worker")
def test_dimension_mapping_filename_uses_output_suffix(monkeypatch):
    url, response = _response("single", "csv")

    _output, mapping_output = _run_concat(
        monkeypatch,
        {url: response},
        {"urls": url, "output_suffix": "run42"},
    )

    assert mapping_output.file_name == "concatenated_run42_dimension_mapping.json"
