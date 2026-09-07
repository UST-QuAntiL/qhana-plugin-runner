# Co-locate plugin tests with plugin code

* Status: [accepted]
* Deciders [Fabian Bühler, EnPro 2026 Team]
* Date: [2026-04-27]

## Context and Problem Statement

All tests in this repository have lived under `tests/` at the repo root, and `pyproject.toml` restricted pytest collection to that directory (`testpaths = ["tests"]`). Plugins under `plugins/` and `stable_plugins/` therefore had no place for their own tests next to the code.

How should the project organise tests so that plugin authors can place tests next to the code they exercise, while still reusing the existing `task_data` fixture and `tests/utils.py` helpers?

## Decision Drivers

* Tests should live close to the code they test.
* Shared setup (the Flask plus DB `task_data` fixture, helpers in `tests/utils.py`) should not be duplicated per plugin.
* The existing relative-import contract for plugin source files, enforced by `tests/test_plugin_imports.py`, must keep working unchanged.
* Test module names can collide across plugins (for example, multiple plugins each with `tests/test_routes.py`). The chosen mechanism must support this. 
* The change should be small: no new build system, no per-plugin `pyproject.toml`, and no breakage to the existing CI command.

## Considered Options

* **Option 1, mirror the plugin tree under `tests/`.** Keep `testpaths = ["tests"]`. Create `tests/plugins/<plugin>/test_*.py` paths that mirror the plugin layout.
* **Option 2, co-locate tests next to plugins; share fixtures via a root `conftest.py`.** Expand `testpaths` to include the plugin trees, move shared fixtures to a repo-root `conftest.py`, put `tests/` on the python path so helpers are importable, and switch pytest to `--import-mode=importlib` to handle name collisions.
* **Option 3, make each plugin a fully installed package.** Give every plugin its own `pyproject.toml` with its own `[tool.pytest.ini_options]` and run tests per-package.

## Decision Outcome

Chosen option: **Option 2, co-locate tests next to plugins**, because it satisfies the locality and ownership drivers with the smallest change to existing tooling. The remaining costs (an import-mode switch and an exclusion in the import checker) are small and easy to implement.

The convention is documented for plugin authors in [`docs/testing.rst`](../testing.rst).

### Positive Consequences

* Tests live next to the code they cover. Plugin authors do not have to maintain a parallel directory layout.
* The shared `task_data` fixture and `tests/utils.py` helpers are visible to plugin tests without boilerplate.
* Both nested (`plugins/foo/tests/test_*.py`) and flat (`plugins/foo/test_*.py`) layouts work, so plugins can pick what fits.
* The CI command (`poetry run pytest --cov=qhana_plugin_runner ...`) does not need to change. The expanded `testpaths` makes plugin tests discoverable automatically.

### Negative Consequences

* Pytest's import mode changes from the default `prepend` to `importlib`. This is a behaviour change worth being aware of when debugging collection issues, even though it is the mode currently [recommended](https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html#choosing-an-import-mode) by pytest for new code.
* The existing typo `tests/conftests.py` (which prevents fixture auto-discovery and forces `from conftests import task_data` workarounds) must be fixed to `conftest.py` and moved to the repo root for the shared fixtures to be visible to plugin tests.

## Pros and Cons of the Options

### Option 1, mirror tree under `tests/`

* Good, because it requires no configuration change.
* Good, because it keeps the import checker untouched.
* Bad, because tests live away from the code they cover, which hurts discoverability and lets tests drift when plugins move.
* Bad, because the mirrored directory structure has to be maintained by hand and tends to rot.

### Option 2, co-locate, share fixtures via root `conftest.py` (chosen)

* Good, because tests sit next to the plugin they exercise so ownership is unambiguous.
* Good, because shared setup stays in one root `conftest.py` plus `tests/utils.py`.
* Good, because both nested and sibling layouts are supported with one config.
* Bad, because it requires switching to `--import-mode=importlib` to avoid module-name collisions.
* Bad, because the plugin-import checker (`tests/test_plugin_imports.py`) needs an exclusion so it skips test files when walking plugin trees.

### Option 3, per-plugin package with its own `pyproject.toml`

* Good, because it gives strong isolation. Each plugin can pin its own pytest plugins, fixtures, and runtime deps.
* Bad, because it is heavier than the problem requires. Each plugin need a separate project setup.
* Bad, because shared fixtures and helpers would have to be republished as an installable test-utilities package.

## Links

* Refines: [ADR-0007, Plugins must provide metadata](0007-plugins-must-provide-metadata.md)
* Implementation will touch `pyproject.toml` (pytest config), the root `conftest.py` (new, replacing `tests/conftests.py`), and `tests/test_plugin_imports.py` (exclude test files from plugin-import checks).
* Developer-facing documentation: [`docs/testing.rst`](../testing.rst)

---

> Note: drafted with the help of Claude Opus 4.7.

<!-- markdownlint-disable-file MD013 -->
