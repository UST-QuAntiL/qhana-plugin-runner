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

"""Utilities for unit tests."""

import io
import json
import zipfile
from typing import Any, Dict, Optional, Sequence

from celery import Task

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile


class MockResponse:
    """Minimal ``requests.Response`` stand-in backed by an in-memory payload.

    Implements just the surface used by ``open_url`` consumers in plugins:
    ``headers``/``url`` for mimetype and filename detection, ``json``/
    ``iter_lines`` for ``load_entities``, and ``content`` for the zip reader.
    """

    def __init__(
        self,
        url: str,
        content_type: str,
        *,
        text: Optional[str] = None,
        content: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.url = url
        self.headers = {"Content-Type": content_type}
        if headers:
            self.headers.update(headers)
        self._text = text
        if content is not None:
            self.content = content
        else:
            self.content = b"" if text is None else text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def json(self):
        return json.loads(self._text)

    def iter_lines(self, decode_unicode: bool = False, **kwargs):
        # ``requests.iter_lines`` yields lines without their terminators, which
        # matches ``str.splitlines`` and keeps csv and json inputs consistent.
        yield from self._text.splitlines()

    @classmethod
    def from_zip(cls, url: str, members: Dict[str, str]) -> "MockResponse":
        """Build a response whose ``content`` is a zip of ``{name: text}``."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in members.items():
                archive.writestr(name, text)
        return cls(url, "application/zip", content=buffer.getvalue())


def run_plugin_task(
    monkeypatch,  # TODO can this be typed?
    task: Task,
    plugin_module: str,
    url_to_response: Dict[str, MockResponse],
    params: dict,
    *,
    expected_result: str = "Result stored in file",
    timeout: int = 30,
) -> TaskFile:
    """Run a plugin calculation task against in-memory responses.

    Persists a ``ProcessingTask`` with ``params``, patches every ``open_url``
    reference the task reaches (the plugin module given by ``plugin_module``,
    the zip reader, and ``requests``) to serve ``url_to_response``, runs the
    task, and returns the task id.

    Args:
        task: the (cast) Celery task to run, e.g. a plugin ``calculation_task``.
        plugin_module: import path of the plugin module that imported
            ``open_url`` by name, e.g. ``"one_hot_encoding"``.
    """
    db_task = ProcessingTask(
        task_name=task.name,
        parameters=json.dumps(params),
    )
    db_task.save(commit=True)

    def mock_open_url(url, *args, **kwargs):
        return url_to_response[url]

    monkeypatch.setattr(f"{plugin_module}.open_url", mock_open_url)
    monkeypatch.setattr(
        "qhana_plugin_runner.plugin_utils.zip_utils.open_url", mock_open_url
    )
    monkeypatch.setattr("qhana_plugin_runner.requests.open_url", mock_open_url)

    result = task.apply_async(kwargs={"db_id": db_task.id}).get(timeout=timeout)
    assert result == expected_result

    DB.session.expire_all()

    task = ProcessingTask.get_by_id(db_task.id)
    assert task is not None
    assert len(task.outputs) == 1
    return task.outputs[0]


def assert_sequence_equals(expected: Sequence[Any], actual: Sequence[Any]):
    """Assert that two sequences contain matching elements."""
    assert len(actual) == len(
        expected
    ), f"Sequences have different sizes, expected length {len(expected)} but got legth {len(actual)}"
    for index, pair in enumerate(zip(actual, expected)):
        actual_item, expected_item = pair
        assert (
            actual_item == expected_item
        ), f"Pair {index} is not equal. Expected {expected_item} but got {actual_item}"


def assert_sequence_partial_equals(
    expected: Sequence[Any], actual: Sequence[Any], attributes_to_test: Sequence[str]
):
    """Assert that the elements in a sequence match in all attributes defined by ``attributes_to_test``.

    The elements in the list can be dicts or namedtuples.
    """
    assert len(actual) == len(
        expected
    ), f"Sequences have different sizes, expected length {len(expected)} but got legth {len(actual)}"
    for index, pair in enumerate(zip(actual, expected)):
        actual_item, expected_item = pair
        for attr in attributes_to_test:
            if isinstance(actual_item, dict):
                actual_value = actual_item.get(attr)
            else:
                actual_value = getattr(actual_item, attr)
            if isinstance(expected_item, dict):
                expected_value = expected_item.get(attr)
            else:
                expected_value = getattr(expected_item, attr)
            assert (
                expected_value == actual_value
            ), f"Attribute '{attr}' of pair {index} is not equal ({expected_value}!={actual_value}). Expected {expected_item} but got {actual_item}"
