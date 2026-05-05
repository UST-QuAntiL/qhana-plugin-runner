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

"""Example tests showing how plugin Celery tasks can be tested.

Strategy taken from the Celery testing guide
(https://docs.celeryq.dev/en/stable/userguide/testing.html):

``celery.contrib.testing.worker.start_worker`` spins up a real
in-process worker thread consuming from an in-memory broker. This
exercises the full ``apply_async`` -> broker -> worker round-trip,
matching how tasks run in production.

Plugin authors can copy the fixtures into their plugin's own
``tests/`` directory and import their plugin's tasks at module level
so Celery picks up the registration.
"""

from logging import INFO

import pytest
from sqlalchemy.pool import StaticPool

from qhana_plugin_runner import create_app
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db import DB
from qhana_plugin_runner.db.cli import create_db_function
from qhana_plugin_runner.db.models.tasks import ProcessingTask

# ---------------------------------------------------------------------------
# Example tasks under test
# ---------------------------------------------------------------------------


@CELERY.task(name="tests.example.add")
def add(x: int, y: int) -> int:
    """Pure task with no Flask context and no DB."""
    return x + y


@CELERY.task(name="tests.example.boom")
def boom() -> None:
    """Task that always raises, used to test error propagation."""
    raise ValueError("boom")


@CELERY.task(name="tests.example.write_task_data", bind=True)
def write_task_data(self, db_id: int, payload: dict) -> int:
    """Task that mutates a ProcessingTask row.

    Mirrors the typical plugin pattern: load a ProcessingTask by id,
    write results into ``data``/``parameters``, persist, return the id.
    The FlaskTask base class wraps the call in an app context so DB
    access works exactly like in production.
    """
    task = ProcessingTask.get_by_id(db_id)
    task.data = payload
    task.save(commit=True)
    return db_id


# ---------------------------------------------------------------------------
# Fixtures: in-process worker on a memory broker
# ---------------------------------------------------------------------------

_TEST_CONFIG = {
    "SECRET_KEY": "test",
    "DEBUG": False,
    "TESTING": True,
    "JSON_SORT_KEYS": True,
    "JSONIFY_PRETTYPRINT_REGULAR": False,
    "DEFAULT_LOG_SEVERITY": INFO,
    "DEFAULT_LOG_FORMAT_STYLE": "{",
    "DEFAULT_LOG_FORMAT": "{message}",
    "DEFAULT_FILE_STORE": "local_filesystem",
    "FILE_STORE_ROOT_PATH": "files",
    "OPENAPI_VERSION": "3.0.2",
    "OPENAPI_JSON_PATH": "api-spec.json",
    "OPENAPI_URL_PREFIX": "",
    "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
    # StaticPool keeps a single connection alive across threads so the
    # in-memory SQLite database is visible from both the test thread and
    # the worker thread.
    "SQLALCHEMY_ENGINE_OPTIONS": {
        "connect_args": {"check_same_thread": False},
        "poolclass": StaticPool,
    },
    "CELERY": {
        "task_default_queue": "qhana_plugin_runner",
        "broker_url": "memory://",
        "result_backend": "cache+memory://",
        "task_always_eager": False,
        "broker_connection_retry_on_startup": True,
    },
}


@pytest.fixture(scope="module")
def broker_app():
    """App configured with a real Celery broker (in-memory)."""
    app = create_app(dict(_TEST_CONFIG), silent_log=True)
    with app.app_context():
        create_db_function(app)
        yield app


@pytest.fixture(scope="module")
def celery_worker(broker_app):
    """Start an in-process Celery worker thread for the test module.

    ``broker_app`` is required so the CELERY singleton is reconfigured
    against the memory broker before the worker boots. The fixture
    is module-scoped because spinning the worker up and down per test
    is slow.
    """
    from celery.contrib.testing.worker import start_worker

    with start_worker(
        CELERY,
        pool="solo",
        perform_ping_check=False,
        shutdown_timeout=10,
    ) as worker:
        yield worker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_worker_returns_value(celery_worker):
    async_result = add.apply_async(args=(7, 35))
    assert async_result.get(timeout=10) == 42


def test_worker_propagates_exception(celery_worker):
    async_result = boom.apply_async()
    with pytest.raises(ValueError, match="boom"):
        async_result.get(timeout=10)


def test_worker_db_write(broker_app, celery_worker):
    task = ProcessingTask(task_name="worker-example")
    task.save(commit=True)
    db_id = task.id

    returned_id = write_task_data.apply_async(
        args=(db_id, {"answer": 42}),
    ).get(timeout=10)

    assert returned_id == db_id
    # Worker committed in a separate session; expire the test session
    # so the next query reads fresh state from the shared connection
    # instead of returning the cached identity-mapped instance.
    DB.session.expire_all()
    reloaded = ProcessingTask.get_by_id(db_id)
    assert reloaded.data == {"answer": 42}
