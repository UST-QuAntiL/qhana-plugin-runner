# Copyright 2022 QHAna plugin runner contributors.
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

"""Tests for the db module of the plugin_utils."""

from logging import INFO

import pytest
from flask import Flask
from sqlalchemy.pool import StaticPool

from qhana_plugin_runner import create_app
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.cli import create_db_function
from qhana_plugin_runner.db.models.tasks import ProcessingTask

MODULE_NAME = "qhana_plugin_runner"


DEFAULT_TEST_CONFIG = {
    "SECRET_KEY": "test",
    "DEBUG": False,
    "TESTING": True,
    "JSON_SORT_KEYS": True,
    "JSONIFY_PRETTYPRINT_REGULAR": False,
    "DEFAULT_LOG_SEVERITY": INFO,
    "DEFAULT_LOG_FORMAT_STYLE": "{",
    "DEFAULT_LOG_FORMAT": "{asctime} [{levelname:^7}] [{module:<30}] {message}    <{funcName}, {lineno}; {pathname}>",
    "DEFAULT_FILE_STORE": "local_filesystem",
    "FILE_STORE_ROOT_PATH": "files",
    "OPENAPI_VERSION": "3.0.2",
    "OPENAPI_JSON_PATH": "api-spec.json",
    "OPENAPI_URL_PREFIX": "",
    # ``SERVER_NAME`` lets ``flask.url_for`` build URLs without a request
    # context, which the route-level tests in plugin test suites rely on.
    "SERVER_NAME": "localhost.localdomain",
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


@pytest.fixture(scope="function")
def task_data():
    test_config = {}
    test_config.update(DEFAULT_TEST_CONFIG)
    test_config.update({"SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})

    app = create_app(test_config)
    with app.app_context():
        create_db_function(app)
        task_data = ProcessingTask(task_name="test-data")
        task_data.save(commit=True)
        yield task_data


@pytest.fixture(scope="module")
def app():
    """Flask app with the plugin runner and all configured plugins loaded.

    ``create_app`` discovers plugins from ``PLUGIN_FOLDERS`` (set in
    ``.flaskenv``), so every plugin blueprint is registered automatically
    once this fixture runs. Module-scoped to amortise the boot cost across
    test cases in a file.
    """
    test_config = dict(DEFAULT_TEST_CONFIG)
    test_config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"

    flask_app = create_app(test_config)
    with flask_app.app_context():
        create_db_function(flask_app)
        yield flask_app


@pytest.fixture()
def client(app: Flask):
    """Flask test client bound to the plugin-runner app."""
    return app.test_client()


@pytest.fixture(scope="module")
def broker_app():
    """App configured with a real Celery broker (in-memory)."""
    app = create_app(dict(DEFAULT_TEST_CONFIG), silent_log=True)
    with app.app_context():
        create_db_function(app)
        yield app


@pytest.fixture(scope="module")
def celery_worker():
    """Start an in-process Celery worker thread for the test module.

    ``broker_app`` is required so the CELERY singleton is reconfigured
    against the memory broker before the worker boots. The fixture
    is module-scoped because spinning the worker up and down per test
    is slow.
    """
    from celery.contrib.testing.worker import start_worker

    from qhana_plugin_runner.celery import CELERY

    with start_worker(  # pyright: ignore[reportGeneralTypeIssues]
        CELERY,
        pool="solo",
        perform_ping_check=False,
        shutdown_timeout=10,
    ) as worker:
        yield worker
