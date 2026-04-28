# Celery Task Testing Strategy

* Status: [proposed]
* Deciders: [Fabian Bühler, EnPro 2026 Team]
* Date: [2026-04-28]

## Context and Problem Statement

Plugins use Celery to execute long running tasks (see
[ADR 0006](0006-use-celery-task-queue.md)). The project currently has no
established way to write automated tests for these tasks. The runner is
deployed with a Redis broker in production, but tests should not require
Redis or Docker so they can run locally and in GitHub Actions CI without
additional infrastructure.

The Celery testing guide
(<https://docs.celeryq.dev/en/stable/userguide/testing.html>) describes
three supported testing approaches. We need to pick one as the recommended
pattern for plugin authors and add an example test that demonstrates it.

## Decision Drivers

* Tests must run in CI without Docker, Redis, or other external services.
* Tests should exercise the same code path as production, including the
  broker, the result backend, and worker side serialization, so that bugs
  related to task registration, argument serialization, and result
  retrieval are caught.
* Plugin authors should be able to copy a small fixture set into their
  plugin and start writing task tests without learning a new framework.
* Test runtime should be acceptable for a pre-commit or PR check.

## Considered Options

* `task_always_eager` configuration setting
* `pytest-celery` plugin
* `celery.contrib.testing.worker.start_worker` with an in-memory broker

## Decision Outcome

Chosen option: **`celery.contrib.testing.worker.start_worker` with an
in-memory broker**, because it is the only option that exercises the full
`apply_async` to broker to worker round-trip while still running entirely
in-process with no external dependencies. An example implementation lives
in `tests/test_celery_example.py`.

The fixture starts a real Celery worker thread inside the test process,
configured against a `memory://` broker and a `cache+memory://` result
backend. Tasks are registered against the existing `CELERY` singleton at
import time, the same way they are in production.

### Positive Consequences

* Task registration, argument serialization, result serialization, and
  worker side error handling are all exercised by the test.
* No Redis, no Docker, no network. Tests run in plain GitHub Actions
  runners with only the existing Python dependencies.
* Plugin authors can copy the fixtures from `tests/test_celery_example.py`
  into their plugin's own `tests/` directory and import the plugin's
  tasks at module level.

### Negative Consequences

* The worker runs in a separate thread, so tests that touch the database
  must use a thread-safe SQLite configuration (`StaticPool` plus
  `check_same_thread=False`) and call `DB.session.expire_all()` before
  re-reading rows that the worker mutated.
* The in-memory broker is not a perfect stand-in for Redis. Behavior that
  depends on broker specific features (visibility timeouts, persistence,
  priorities) is not covered.

## Pros and Cons of the Options

### `task_always_eager`

Sets the Celery configuration option `task_always_eager = True` so that
calls to `delay()` or `apply_async()` execute the task synchronously in
the calling thread and return an `EagerResult`.

* Good, because it requires no fixtures and runs instantly.
* Good, because debugging is easy since the task runs in the test thread.
* Bad, because the Celery testing guide explicitly warns against this
  approach: "By definition this is not a unit test." It bypasses the
  broker, the worker, and the serialization layer, so the code path
  under test does not match production.
* Bad, because bugs in task registration, argument serialization, and
  result handling are not caught.

### `pytest-celery`

Third party pytest plugin that provides Celery fixtures and matrix style
testing across broker and backend combinations.

* Good, because it exposes a rich fixture API and supports parametrising
  over multiple broker and backend combinations.
* Bad, because the current version of `pytest-celery` requires Docker to
  start broker and backend containers. CI runners that do not provide a
  Docker daemon (including the default GitHub Actions Python images when
  used without `services:`) cannot run the suite.
* Bad, because it adds a dependency that other parts of the project do
  not use.

### `celery.contrib.testing.worker.start_worker` with an in-memory broker

Documented testing helper that starts a Celery worker thread inside the
test process. Combined with `broker_url = "memory://"` and
`result_backend = "cache+memory://"` it runs entirely in-process.

* Good, because it ships with Celery itself, so no extra dependency.
* Good, because it exercises the real broker dispatch and result backend
  paths.
* Good, because it works in any environment that can run Python, which
  includes plain GitHub Actions runners.
* Bad, because the worker runs in a thread and shared state (in-memory
  SQLite, SQLAlchemy session caches) needs explicit handling.
* Bad, because the worker fixture has a small startup cost.

## Links

* Refines [ADR 0006](0006-use-celery-task-queue.md)
* Celery testing guide: <https://docs.celeryq.dev/en/stable/userguide/testing.html>
* Example implementation: `tests/test_celery_example.py`

---

> Note: drafted with the help of Claude Opus 4.7.

<!-- markdownlint-disable-file MD013 -->
