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
from typing import Any, Dict, List, Optional, Sequence, Union

import requests
from celery import Task

from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile


class MockResponse(requests.Response):
    """``requests.Response`` backed by an in-memory payload.

    Subclasses the real response class so ``isinstance`` checks in helpers
    like ``retrieve_filename`` accept it. Content access (``content``,
    ``text``, ``json``, ``iter_lines``) uses the inherited implementations.
    """

    def __init__(
        self,
        url: str,
        content_type: str,
        *,
        text: Optional[str] = None,
        content: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
        status_code: int = 200,
        json_data: Optional[Any] = None,
    ):
        super().__init__()
        self.url = url
        self.status_code = status_code
        self.encoding = "utf-8"
        self.headers["Content-Type"] = content_type
        if headers:
            self.headers.update(headers)
        if json_data is not None and content is None and text is None:
            text = json.dumps(json_data)
        if content is not None:
            self._content = content
        else:
            self._content = b"" if text is None else text.encode("utf-8")
        # Mark the content as consumed so ``close`` and ``iter_content`` do
        # not touch the (absent) underlying connection.
        self._content_consumed = True

    @classmethod
    def from_zip(cls, url: str, members: Dict[str, str]) -> "MockResponse":
        """Build a response whose ``content`` is a zip of ``{name: text}``."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in members.items():
                archive.writestr(name, text)
        return cls(url, "application/zip", content=buffer.getvalue())


def run_task(task: Task, *, timeout: int = 30, **task_kwargs) -> Any:
    """Dispatch a Celery task on the running worker and return its result.

    Sends ``task_kwargs`` to the task through the broker, waits up to
    ``timeout`` seconds for the result, and expires the test session so rows
    mutated by the worker thread are re-read from the database.

    Requires the ``celery_worker`` fixture so a worker consumes the message.
    """
    result = task.apply_async(kwargs=task_kwargs).get(timeout=timeout)
    DB.session.expire_all()
    return result


def mock_task_dispatch(monkeypatch):
    """Turn Celery task dispatch into a no-op for tests without a worker.

    Patches ``apply_async`` on both task signatures and tasks so route-level
    tests can exercise the HTTP layer without enqueuing messages on the shared
    broker. Returns a dummy result object carrying an ``id``.
    """
    from celery.app.task import Task as CeleryTask
    from celery.canvas import Signature

    class _DummyResult:
        id = "mock-task-id"

    def _noop(self, *args, **kwargs):
        return _DummyResult()

    monkeypatch.setattr(Signature, "apply_async", _noop)
    monkeypatch.setattr(CeleryTask, "apply_async", _noop)


def run_plugin_task_outputs(
    monkeypatch,  # TODO can this be typed?
    task: Task,
    plugin_module: str,
    url_to_response: Dict[str, MockResponse],
    params: Union[dict, str],
    *,
    expected_result: str = "Result stored in file",
    timeout: int = 30,
) -> List[TaskFile]:
    """Run a plugin calculation task against in-memory responses.

    Persists a ``ProcessingTask`` with ``params``, patches every ``open_url``
    reference the task reaches (the plugin module given by ``plugin_module``,
    the zip reader, and ``requests``) to serve ``url_to_response``, runs the
    task, and returns all persisted output files.

    Use :py:func:`run_plugin_task` instead when the task produces exactly one
    output file.

    Args:
        task: the (cast) Celery task to run, e.g. a plugin ``calculation_task``.
        plugin_module: import path of the plugin module that imported
            ``open_url`` by name, e.g. ``"one_hot_encoding"``.
        params: task parameters, stored as JSON when given as a dict and
            unchanged when given as a string.
    """
    db_task = ProcessingTask(
        task_name=task.name,
        parameters=params if isinstance(params, str) else json.dumps(params),
    )
    db_task.save(commit=True)

    def mock_open_url(url, *args, **kwargs):
        return url_to_response[url]

    monkeypatch.setattr(f"{plugin_module}.open_url", mock_open_url)
    monkeypatch.setattr(
        "qhana_plugin_runner.plugin_utils.zip_utils.open_url", mock_open_url
    )
    monkeypatch.setattr("qhana_plugin_runner.requests.open_url", mock_open_url)

    result = run_task(task, db_id=db_task.id, timeout=timeout)
    assert result == expected_result

    task = ProcessingTask.get_by_id(db_task.id)
    assert task is not None
    return list(task.outputs)


def run_plugin_task(
    monkeypatch,  # TODO can this be typed?
    task: Task,
    plugin_module: str,
    url_to_response: Dict[str, MockResponse],
    params: Union[dict, str],
    *,
    expected_result: str = "Result stored in file",
    timeout: int = 30,
) -> TaskFile:
    """Run a plugin calculation task that produces exactly one output file.

    Thin wrapper around :py:func:`run_plugin_task_outputs` that asserts a single
    output and returns it.
    """
    outputs = run_plugin_task_outputs(
        monkeypatch,
        task,
        plugin_module,
        url_to_response,
        params,
        expected_result=expected_result,
        timeout=timeout,
    )
    assert len(outputs) == 1
    return outputs[0]


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
