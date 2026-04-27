# Testing

> Note: this file is a draft. It still needs to be adapted into a proper documentation page (Sphinx integration, cross-links from `index.rst`, and review of style/structure).

This project uses [pytest](https://docs.pytest.org/) with [hypothesis](https://hypothesis.readthedocs.io/) for property-based tests. Tests can live in two places:

- `tests/` at the repo root, for cross-cutting tests of the `qhana_plugin_runner` package and for shared setup that is reused everywhere.
- Next to the plugin, as `plugins/<plugin>/tests/test_*.py` (nested) or `plugins/<plugin>/test_*.py` (sibling). Both are picked up by pytest. Use whichever layout fits the plugin.

The convention is recorded in [ADR-0018: Co-locate plugin tests with plugin code](adr/0018-co-locate-plugin-tests.md).

## Shared setup

Plugin-local tests can use the shared setup without extra imports for fixtures, and with a single import for helpers.

### Fixtures (root `conftest.py`)

Cross-cutting fixtures live in `conftest.py` at the repo root. Pytest discovers `conftest.py` from every ancestor directory of a test file, so any test under `tests/`, `plugins/**`, or `stable_plugins/**` can request these fixtures by parameter name without importing them.

The main shared fixture is `task_data`. It creates the Flask app with an in-memory SQLite database and yields a saved `ProcessingTask`:

```python
def test_my_plugin_persists_state(task_data):
    task_data.data = {"foo": "bar"}
    task_data.save(commit=True)
    assert task_data.data["foo"] == "bar"
```

### Helpers (`tests/utils.py`)

Reusable assertions and utilities live in `tests/utils.py`. Because `pyproject.toml` sets `pythonpath = ["tests"]`, they can be imported as a top-level module from anywhere:

```python
from utils import assert_sequence_equals
```

## Pytest configuration

The relevant settings in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests", "plugins", "stable_plugins"]
pythonpath = ["tests"]
addopts = "--import-mode=importlib"
```

- `testpaths` lists the directories pytest walks for test collection.
- `pythonpath = ["tests"]` puts `tests/` on `sys.path` so `from utils import ...` resolves.
- `--import-mode=importlib` is needed because two plugins may each have a file with the same name (for example `tests/test_routes.py`). The default `prepend` mode raises `ImportError: import file mismatch` in that case. `importlib` mode imports each test file under a unique synthetic module name, and does not require adding `__init__.py` files inside plugin packages.

## Plugin source vs. plugin tests

`tests/test_plugin_imports.py` enforces that plugin **source** files use only relative imports, so plugins remain portable, lazy-loadable packages.

That rule does not apply to plugin **test** files. Anything inside a plugin's `tests/` subdirectory, or matching `test_*.py` or `conftest.py`, is skipped by the import checker. Plugin tests can use absolute imports of `pytest`, `hypothesis`, `qhana_plugin_runner.…`, and so on.

| File                                                   | Imports allowed                                                                                                      |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `plugins/foo/__init__.py`, `plugins/foo/*.py`          | Relative only (`from .routes import ...`)                                                                            |
| `plugins/foo/tests/test_*.py`, `plugins/foo/test_*.py` | Any (`import pytest`, `from qhana_plugin_runner.db.models.tasks import ProcessingTask`, `from utils import ...`, ...) |

## Example: a minimal plugin test

```python
# plugins/sql_loader/tests/test_smoke.py
from utils import assert_sequence_equals

from qhana_plugin_runner.db.models.tasks import ProcessingTask


def test_task_data_fixture(task_data: ProcessingTask):
    assert task_data.task_name == "test-data"
    assert task_data.data == {}


def test_helpers_are_importable():
    assert_sequence_equals([1, 2, 3], [1, 2, 3])
```

No `conftest.py`, no path manipulation, and no fixture import is required.

## Running tests

```bash
# All tests (repo + plugins)
poetry run pytest

# Just one plugin
poetry run pytest plugins/sql_loader

# Just the repo-wide tests
poetry run pytest tests

# With coverage (matches CI)
poetry run pytest --cov=qhana_plugin_runner --cov-report=term

# Re-run only the tests that failed last time
poetry run pytest --last-failed
```

---

> Note: drafted with the help of Claude Opus 4.7.
