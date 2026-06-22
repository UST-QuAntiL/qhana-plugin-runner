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

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from vector_concat.tasks import calculation_task


# TODO: could be extracted to utilities file in root test folder if this is usefull for other plugin tests
class _MockResponse:
    def __init__(self, content_type: str, *, json_data=None, csv_lines=None):
        self.headers = {"Content-Type": content_type}
        self._json_data = json_data
        self._csv_lines = csv_lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def json(self):
        return self._json_data

    def iter_lines(self, decode_unicode: bool = False, **kwargs):
        yield from self._csv_lines or ()


def _csv_response(header, rows):
    lines = [",".join(header)]
    lines += [",".join(str(v) for v in row) for row in rows]
    return _MockResponse("text/csv", csv_lines=lines)


def _json_response(entities):
    return _MockResponse("application/json", json_data=entities)


def _run_task(monkeypatch, url_to_response: dict, params: dict):
    db_task = ProcessingTask(
        task_name=calculation_task.name,
        parameters=json.dumps(params),
    )
    db_task.save(commit=True)

    def mock_open_url(url, *args, **kwargs):
        return url_to_response[url]

    monkeypatch.setattr("vector_concat.tasks.open_url", mock_open_url)

    result = calculation_task.apply(kwargs={"db_id": db_task.id}).get()
    assert result == "Result stored in file"

    DB.session.expire_all()
    return db_task.id


def _single_output(db_id: int):
    task = ProcessingTask.get_by_id(db_id)
    assert task is not None
    assert len(task.outputs) == 1
    return task.outputs[0]


def _read_json(file_info):
    with open(file_info.file_storage_data, "r") as fh:
        return json.load(fh)


def _read_csv(file_info):
    with open(file_info.file_storage_data, "r", newline="") as fh:
        return list(csv.DictReader(fh))


def test_concat_two_csv_inputs_produces_combined_csv(app, monkeypatch):
    url_a, url_b = "http://example.com/a.csv", "http://example.com/b.csv"
    responses = {
        url_a: _csv_response(["ID", "href", "dim0", "dim1"], [("e1", "h1", 1, 2)]),
        url_b: _csv_response(["ID", "href", "dim0", "dim1"], [("e1", "h1", 3, 4)]),
    }
    db_id = _run_task(
        monkeypatch, responses, {"urls": f"{url_a}\n{url_b}", "output_format": "csv"}
    )

    output = _single_output(db_id)
    assert output.file_name == "concatenated.csv"
    assert output.file_type == "entity/vector"
    assert output.mimetype == "text/csv"

    rows = _read_csv(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["ID"] == "e1"
    assert row["href"] == "h1"
    assert [row["dim0"], row["dim1"], row["dim2"], row["dim3"]] == ["1", "2", "3", "4"]


def test_concat_two_json_inputs_produces_combined_json(app, monkeypatch):
    url_a, url_b = "http://example.com/a.json", "http://example.com/b.json"
    responses = {
        url_a: _json_response([{"ID": "e1", "href": "h1", "dim0": 1, "dim1": 2}]),
        url_b: _json_response([{"ID": "e1", "href": "h1", "dim0": 3, "dim1": 4}]),
    }
    db_id = _run_task(
        monkeypatch, responses, {"urls": f"{url_a}\n{url_b}", "output_format": "json"}
    )

    output = _single_output(db_id)
    assert output.file_name == "concatenated.json"
    assert output.file_type == "entity/vector"
    assert output.mimetype == "application/json"

    entities = _read_json(output)
    assert len(entities) == 1
    entity = entities[0]
    assert entity["ID"] == "e1"
    assert entity["href"] == "h1"
    assert [entity["dim0"], entity["dim1"], entity["dim2"], entity["dim3"]] == [
        1,
        2,
        3,
        4,
    ]


def test_output_suffix_appends_to_csv_filename(app, monkeypatch):
    url_a, url_b = "http://example.com/a.csv", "http://example.com/b.csv"
    responses = {
        url_a: _csv_response(["ID", "href", "dim0"], [("e1", "h1", 1)]),
        url_b: _csv_response(["ID", "href", "dim0"], [("e1", "h1", 2)]),
    }
    db_id = _run_task(
        monkeypatch,
        responses,
        {"urls": f"{url_a}\n{url_b}", "output_format": "csv", "output_suffix": "run42"},
    )

    output = _single_output(db_id)
    assert output.file_name == "concatenated_run42.csv"


def test_output_suffix_appends_to_json_filename(app, monkeypatch):
    url_a, url_b = "http://example.com/a.json", "http://example.com/b.json"
    responses = {
        url_a: _json_response([{"ID": "e1", "href": "h1", "dim0": 1}]),
        url_b: _json_response([{"ID": "e1", "href": "h1", "dim0": 2}]),
    }
    db_id = _run_task(
        monkeypatch,
        responses,
        {"urls": f"{url_a}\n{url_b}", "output_format": "json", "output_suffix": "run42"},
    )

    output = _single_output(db_id)
    assert output.file_name == "concatenated_run42.json"


def test_empty_output_suffix_keeps_default_filename(app, monkeypatch):
    url = "http://example.com/a.csv"
    responses = {url: _csv_response(["ID", "href", "dim0"], [("e1", "h1", 5)])}
    db_id = _run_task(monkeypatch, responses, {"urls": url, "output_suffix": "   "})

    output = _single_output(db_id)
    assert output.file_name == "concatenated.csv"


def test_output_format_defaults_to_csv(app, monkeypatch):
    """When ``output_format`` is absent the task falls back to CSV output."""
    url = "http://example.com/a.csv"
    responses = {url: _csv_response(["ID", "href", "dim0"], [("e1", "h1", 5)])}
    db_id = _run_task(monkeypatch, responses, {"urls": url})

    output = _single_output(db_id)
    assert output.file_name == "concatenated.csv"
    assert output.mimetype == "text/csv"


def test_mismatched_ids_raise(app, monkeypatch):
    url_a, url_b = "http://example.com/a.csv", "http://example.com/b.csv"
    responses = {
        url_a: _csv_response(["ID", "href", "dim0"], [("e1", "h1", 1)]),
        url_b: _csv_response(["ID", "href", "dim0"], [("e2", "h1", 2)]),
    }
    with pytest.raises(AssertionError):
        _run_task(
            monkeypatch, responses, {"urls": f"{url_a}\n{url_b}", "output_format": "csv"}
        )


def test_unequal_row_counts_raise(app, monkeypatch):
    """``zip(*entities_list, strict=True)`` rejects inputs of differing length."""
    url_a, url_b = "http://example.com/a.csv", "http://example.com/b.csv"
    responses = {
        url_a: _csv_response(["ID", "href", "dim0"], [("e1", "h1", 1), ("e2", "h2", 2)]),
        url_b: _csv_response(["ID", "href", "dim0"], [("e1", "h1", 3)]),
    }
    with pytest.raises(ValueError):
        _run_task(
            monkeypatch, responses, {"urls": f"{url_a}\n{url_b}", "output_format": "csv"}
        )


def test_missing_db_id_raises(app):
    """Task raises ``KeyError`` when no ``ProcessingTask`` row matches the id."""
    with pytest.raises(KeyError, match="Could not load task data"):
        calculation_task.apply(kwargs={"db_id": 99999}).get()
