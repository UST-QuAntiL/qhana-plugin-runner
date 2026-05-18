Writing Tests
=============

This guide is for contributors writing tests for the QHAna plugin runner core or for individual plugins.
It covers the pytest configuration, the test file locations, the shared fixtures and helpers, the recommended pattern for testing Celery tasks, how to run the suite locally, and how it is executed in CI.

The testing strategy is anchored in two architecture decision records:

* :doc:`adr/0018-celery-task-testing-strategy`. Celery tasks are tested with an in-process worker on an in-memory broker.
* :doc:`adr/0019-co-locate-plugin-tests`. Plugin tests are co-located with the plugin code.

For background on why plugins use Celery in the first place, see :doc:`adr/0006-use-celery-task-queue`.


Pytest setup
------------

The pytest configuration is defined in :source:`pyproject.toml`:

.. code-block:: toml

    [tool.pytest.ini_options]
    testpaths = ["tests", "plugins", "stable_plugins"]
    pythonpath = ["tests"]
    addopts = "--import-mode=importlib"
    python_files = ["test_*.py"]

* ``testpaths`` collects tests from the runner-core ``tests/`` directory and from both plugin trees.
* ``pythonpath = ["tests"]`` puts ``tests/`` on ``sys.path`` so shared helpers can be imported as ``from utils import …``.
* ``addopts = "--import-mode=importlib"`` switches pytest from the default ``prepend`` import mode to ``importlib``. This is required because plugin test modules can collide on names (for example several plugins each shipping their own ``test_routes.py``). See :doc:`adr/0019-co-locate-plugin-tests` for the rationale.
* ``python_files = ["test_*.py"]`` restricts collection to files starting with ``test_``.

The unit tests depend on `pytest <https://docs.pytest.org/>`_ and `hypothesis <https://hypothesis.readthedocs.io/>`_, both pulled in by the dev dependency group.


Test File Locations
-------------------

Test files reside in three valid locations:

* ``tests/`` for runner-core tests.
  Examples: :source:`tests/test_db.py`, :source:`tests/test_entity_marshalling.py`, :source:`tests/test_plugin_imports.py`.
* ``plugins/<name>/`` for plugin tests, in either a nested or a flat layout (see below).
* ``stable_plugins/<theme>/<plugin>/`` for stable-plugin tests, following the same nested-or-flat convention.

A plugin chooses one of two layouts for its tests:

* **Nested layout.** Test files reside in a dedicated ``tests/`` subdirectory within the plugin package:

  .. code-block:: text

      plugins/foo/
      ├── __init__.py
      ├── routes.py
      ├── tasks.py
      └── tests/
          ├── __init__.py
          └── test_routes.py

* **Flat layout.** Test files reside directly next to the source files, prefixed with ``test_``:

  .. code-block:: text

      plugins/bar/
      ├── __init__.py
      ├── routes.py
      ├── tasks.py
      └── test_routes.py

Pytest discovers both layouts. The nested layout suits plugins with many test files or shared fixtures specific to the plugin. The flat layout suits small plugins where one or two test modules sit comfortably alongside the source. Module-name collisions across plugins (for example two plugins each having ``test_routes.py``) are handled by ``--import-mode=importlib``.

.. seealso:: The ADR :doc:`adr/0019-co-locate-plugin-tests` records the reasoning for this layout and the trade-offs against alternatives (mirroring under ``tests/``, per-plugin ``pyproject.toml``).


Shared fixtures and helpers
---------------------------

Pytest **fixtures** are functions that prepare test state (a database row, a temp file, a Flask app) and inject it into tests by parameter name. Pytest runs the setup once per fixture scope (function, module, or session), then tears it down afterwards. They replace the ``setUp``/``tearDown`` boilerplate of class-based test frameworks and let each test declare exactly the dependencies it needs. To use a fixture, name it as a parameter of the test function:

.. code-block:: python

    def test_something(task_data):
        assert task_data.task_name == "test-data"

See the `pytest fixture guide <https://docs.pytest.org/en/stable/how-to/fixtures.html>`_ for the full feature set (scopes, parametrization, ``yield`` teardown, ``autouse``, indirect fixtures).

Pytest auto-discovers a repo-root ``conftest.py`` that provides shared fixtures usable from any test in any of the three test locations. Tests do not need to import these fixtures explicitly. Pytest injects them by parameter name.

The ``task_data`` fixture
~~~~~~~~~~~~~~~~~~~~~~~~~

``task_data`` builds an in-memory SQLite Flask app via :py:func:`~qhana_plugin_runner.create_app`, creates the database schema, and yields a saved :py:class:`~qhana_plugin_runner.db.models.tasks.ProcessingTask`. Use it in any test that needs a real Flask app context plus a ``ProcessingTask`` row to operate on:

.. code-block:: python

    from qhana_plugin_runner.db.models.tasks import ProcessingTask


    def test_task_persists(task_data):
        reloaded = ProcessingTask.get_by_id(task_data.id)
        assert reloaded.task_name == "test-data"

The fixture is function-scoped, so each test gets a fresh database.

The ``app`` and ``client`` fixtures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``app`` is a module-scoped Flask application built via :py:func:`~qhana_plugin_runner.create_app` with the same in-memory SQLite configuration as ``task_data``.
Plugin discovery runs as part of ``create_app``, so every blueprint declared under ``PLUGIN_FOLDERS`` is registered on the returned app.
Module scope amortises the startup cost of plugin discovery across the test cases in a file.
Use ``app`` whenever a test needs the full configured application, an application context, or :py:func:`flask.url_for` without a request context (the test configuration sets ``SERVER_NAME`` so ``url_for`` can build URLs outside a request).

``client`` is a function-scoped :py:meth:`flask.Flask.test_client` bound to ``app``.
It is the standard entry point for HTTP-level tests and removes the need for a plugin-local Flask fixture:

.. code-block:: python

    from http import HTTPStatus
    from flask import url_for


    def test_metadata_endpoint(client):
        response = client.get(url_for("data-creator-v0-1-1.PluginsView"))
        assert response.status_code == HTTPStatus.OK

Both fixtures are defined in the repo-root :source:`conftest.py` and are auto-discovered.

Assertion helpers in ``tests/utils.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:source:`tests/utils.py` holds reusable assertions. Import as ``from utils import …``, which works because ``tests/`` is on the python path.

* ``assert_sequence_equals(expected, actual)``: element-wise equality with index-aware error messages.
* ``assert_sequence_partial_equals(expected, actual, attributes_to_test)``: checks only the listed attributes of dict or namedtuple elements.

Why fixtures?
~~~~~~~~~~~~~

The shared fixtures give plugin authors a real Flask app with a real database without booting Redis, Postgres, or Docker. Tests stay fast, deterministic, and run on plain CI runners with no extra services.


Writing tests for the plugin runner
-----------------------------------

Tests for runner-core code go in ``tests/`` and follow standard pytest patterns. The existing modules are good templates:

* :source:`tests/test_db.py`: minimal fixture usage with the ``task_data`` fixture.
* :source:`tests/test_entity_marshalling.py`: exercises CSV/JSON entity round-trips with ``assert_sequence_equals``.
* :source:`tests/test_plugin_imports.py`: validates the plugin import contract enforced on plugin source files.

Use ``task_data`` whenever a test needs a Flask app context or DB access.


Writing tests for plugins
-------------------------

To test a plugin:

1. Place test files at ``plugins/<name>/tests/test_*.py`` or ``plugins/<name>/test_*.py``.
2. Reuse the shared ``task_data`` fixture and the helpers from ``tests/utils.py``, with no boilerplate and no duplication.
3. Plugin source files **must use relative imports** (this is enforced by ``tests/test_plugin_imports.py`` so plugins remain relocatable, see :doc:`plugins`). Test files are excluded from this check, so plugin tests can use absolute imports.

Test module names can collide across plugins (multiple plugins each having a ``test_routes.py`` is fine). ``--import-mode=importlib`` handles the disambiguation.

Examples
~~~~~~~~

The :source:`stable_plugins/data_synthesis/data_creator/tests/` directory demonstrates the test types described in this guide. It uses the nested layout and relies on the ``client`` fixture from the repo-root :source:`conftest.py`. Each file covers one aspect of the plugin:

* :source:`stable_plugins/data_synthesis/data_creator/tests/test_datasets.py`  

  Pure unit tests and hypothesis property tests for the numpy-based dataset generators in :source:`stable_plugins/data_synthesis/data_creator/backend/datasets.py`. Demonstrates :py:func:`pytest.mark.parametrize` for shape checks across every ``DataTypeEnum`` member, and ``@given`` strategies for invariants (output length, finite values, integer label dtype, label range bounded by ``centers``). No fixtures are required because the generators have no Flask, DB, or Celery dependencies.
* :source:`stable_plugins/data_synthesis/data_creator/tests/test_schemas.py`

  Marshmallow schema tests for ``InputParametersSchema``. Covers the round-trip from JSON payload to ``InputParameters`` dataclass (including the ``camelCase`` rewriting performed by ``MaBaseSchema``), the per-type required-field rules in ``REQUIRED_FIELDS_BY_TYPE``, range validators on ``num_train_points`` / ``noise`` / ``turns`` / ``centers``, and rejection of unknown ``dataset_type`` values. Schemas are pure Python, so these tests also run without a Flask app.
* :source:`stable_plugins/data_synthesis/data_creator/tests/test_routes.py`

  HTTP-level tests using the shared ``client`` fixture. Covers the metadata endpoint (``GET /plugins/<id>/``), the micro frontend form rendering and default values, and the form's behavior on invalid input (the route uses ``validate_errors_as_result=True`` and re-renders rather than returning a 400). The ``/process/`` endpoint enqueues a Celery task and is therefore covered by Celery-aware tests instead, see :doc:`adr/0018-celery-task-testing-strategy`.

* :source:`stable_plugins/data_synthesis/data_creator/tests/test_tasks.py`

  End-to-end Celery tests for the ``calculation_task`` enqueued by ``/process/``. Persists a ``ProcessingTask`` the way ``routes.py`` does, calls ``calculation_task.apply_async`` against the in-memory broker, and asserts on the four output files written by the worker (file names, ``file_type``, ``mimetype``, and JSON payload shape). Also covers the ``centers`` parameter for ``DataTypeEnum.blobs`` and the ``KeyError`` raised when the ``db_id`` does not resolve to a row. Uses the ``broker_app`` and ``celery_worker`` fixtures from the repo-root :source:`conftest.py`, following the pattern described in `Testing Celery tasks`_.


Testing Celery tasks
--------------------

Plugins use Celery for long-running work (see :doc:`adr/0006-use-celery-task-queue`). The recommended testing strategy, set out in :doc:`adr/0018-celery-task-testing-strategy`, is to run a real Celery worker thread inside the test process against an in-memory broker. This exercises the full ``apply_async`` → broker → worker round-trip (including task registration, argument serialization, result serialization, and worker-side error handling) without requiring Redis or Docker.

The fixtures and test config required for this pattern can be found in the repo-root :source:`conftest.py` and are auto-discovered by pytest.
Plugin authors do not need to copy them.
Import the plugin's tasks at module level in the test file so the ``CELERY`` singleton picks up the registration when ``broker_app`` builds the app.

Configuration
~~~~~~~~~~~~~

The Flask + Celery test config in :source:`conftest.py` combines an in-memory SQLite database (with a thread-safe pool) with an in-memory Celery broker:

.. literalinclude:: ../conftest.py
    :language: python
    :lines: 31-62
    :emphasize-lines: 18-24, 25-31

Two parts are critical:

* ``SQLALCHEMY_ENGINE_OPTIONS`` uses ``StaticPool`` and ``check_same_thread=False`` so the in-memory SQLite database is visible from both the test thread and the worker thread.
* The ``CELERY`` block uses ``broker_url = "memory://"`` and ``result_backend = "cache+memory://"`` and keeps ``task_always_eager = False`` so calls actually go through the broker.

Fixtures
~~~~~~~~

Two module-scoped fixtures in :source:`conftest.py` set up the app and the worker thread:

.. literalinclude:: ../conftest.py
    :language: python
    :lines: 103-129

* ``broker_app`` builds the Flask app with the test config and creates the database schema.
* ``celery_worker`` starts a real in-process Celery worker via ``celery.contrib.testing.worker.start_worker``. ``pool="solo"`` keeps the worker single-threaded for simpler debugging. The fixture is module-scoped because spinning the worker up and down per test is slow.

Use both fixtures in every Celery test, either by naming them as parameters or via ``@pytest.mark.usefixtures("broker_app", "celery_worker")`` when the test body does not reference them directly.

Example tests
~~~~~~~~~~~~~

:source:`stable_plugins/data_synthesis/data_creator/tests/test_tasks.py` demonstrates this pattern for a plugin.

.. code-block:: python

    @pytest.mark.usefixtures("broker_app", "celery_worker")
    def test_calculation_task_persists_four_files():
        db_id = _enqueue_processing_task(...)
        result = calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)
        assert result == "Result stored in file"

Errors propagate through the result backend and can be asserted with ``pytest.raises``:

.. code-block:: python

    @pytest.mark.usefixtures("broker_app", "celery_worker")
    def test_calculation_task_missing_db_id_raises():
        async_result = calculation_task.apply_async(kwargs={"db_id": 99999})
        with pytest.raises(KeyError, match="Could not load task data"):
            async_result.get(timeout=30)

A test that exercises a DB-mutating task must expire the test session before re-reading the row, otherwise SQLAlchemy returns the cached identity-mapped instance from before the worker committed:

.. code-block:: python
    :emphasize-lines: 5

    @pytest.mark.usefixtures("broker_app", "celery_worker")
    def test_reads_worker_mutation():
        db_id = _enqueue_processing_task(...)
        calculation_task.apply_async(kwargs={"db_id": db_id}).get(timeout=30)
        DB.session.expire_all()
        task = ProcessingTask.get_by_id(db_id)
        assert task.outputs  # written by the worker thread

Gotchas
~~~~~~~

.. warning::

    * Use ``StaticPool`` with ``check_same_thread=False`` for any in-memory SQLite database that the worker thread will touch. Without this, the test thread and the worker thread see different databases.
    * Call ``DB.session.expire_all()`` before re-reading rows that the worker mutated. The test session caches identity-mapped instances and will otherwise return stale state.
    * Do **not** use ``task_always_eager = True``. It bypasses the broker, the worker, and the serialization layer, so the code path under test does not match production. :doc:`adr/0018-celery-task-testing-strategy` explicitly rejects this option.
    * The in-memory broker does not model Redis-specific behavior (visibility timeouts, persistence, priorities). Tests that depend on those features need a real broker.

.. seealso::

    * :doc:`adr/0018-celery-task-testing-strategy`
    * `Celery testing guide <https://docs.celeryq.dev/en/stable/userguide/testing.html>`_
    * Plugin example: :source:`stable_plugins/data_synthesis/data_creator/tests/test_tasks.py`
    * Fixtures and test config: :source:`conftest.py`


Property-based testing with hypothesis
--------------------------------------

`Hypothesis <https://hypothesis.readthedocs.io/>`_ is a property-based testing library. Instead of writing example-driven assertions, you describe the *property* a function should satisfy and hypothesis generates many inputs to try and falsify it. When it finds a counter-example, it shrinks the input to a minimal failing case before reporting it. This is well-suited for code with structured input domains (entity marshalling, attribute parsers, serialization round-trips), where hand-picking examples tends to miss edge cases.

A round-trip property looks like:

.. code-block:: python

    from hypothesis import given, strategies as st


    @given(st.dictionaries(st.text(), st.integers()))
    def test_roundtrip(data):
        assert deserialize(serialize(data)) == data

Hypothesis is pulled in by the dev dependency group, so no extra setup is needed. ``poetry run pytest --hypothesis-explain`` prints the example-shrinking trail when a property fails, which helps when the minimized counter-example is not self-explanatory.

See the `hypothesis quickstart <https://hypothesis.readthedocs.io/en/latest/quickstart.html>`_ and the `strategies reference <https://hypothesis.readthedocs.io/en/latest/data.html>`_ for the full API.


Running tests
-------------

The full set of pytest commands is documented in the :doc:`readme`. The most-used invocations:

.. code-block:: bash

    # Run the whole suite
    poetry run pytest

    # Run a single test
    poetry run pytest path/to/test_x.py::test_name

    # Re-run only failures from the last run
    poetry run pytest --last-failed

    # Coverage with a terminal summary and an HTML report under htmlcov/
    poetry run pytest -p pytest_cov --cov=qhana_plugin_runner --cov-report=html --cov-report=term


Continuous integration
----------------------

Unit tests run on every push to ``main`` and on every pull request via :source:`.github/workflows/pytest.yml`. The job sets up Python 3.10, installs dependencies with ``poetry install --no-interaction --with dev``, runs ``poetry run pytest --cov=qhana_plugin_runner --cov-report=html --cov-report=term``, and uploads the HTML coverage report as a build artifact. No external services are started. The in-memory SQLite database and the in-memory Celery broker keep the suite self-contained.

A separate workflow at :source:`.github/workflows/integration-tests.yml` runs the full QHAna integration suite (`UST-QuAntiL/qhana-integration-tests <https://github.com/UST-QuAntiL/qhana-integration-tests>`_) on a weekly schedule and on manual dispatch. That workflow exercises the runner against a real broker, registry, backend, and UI. It is out of scope for this guide.


See also
--------

* :doc:`adr/0006-use-celery-task-queue`
* :doc:`adr/0018-celery-task-testing-strategy`
* :doc:`adr/0019-co-locate-plugin-tests`
* :source:`stable_plugins/data_synthesis/data_creator/tests/`, an example covering unit, property-based, schema, and route tests for a stable plugin.
* :doc:`plugins`, the plugin authoring guide.
* `pytest documentation <https://docs.pytest.org/>`_
* `hypothesis documentation <https://hypothesis.readthedocs.io/>`_
* `Celery testing guide <https://docs.celeryq.dev/en/stable/userguide/testing.html>`_
* `Flask testing guide <https://flask.palletsprojects.com/en/stable/testing/>`_
