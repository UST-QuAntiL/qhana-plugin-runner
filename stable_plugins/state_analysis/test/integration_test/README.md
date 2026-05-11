# State analysis — integration tests

These tests hit a running `qhana-plugin-runner` over HTTP. They are **not**
run by the project's normal `pytest` invocation; you launch them
explicitly against a running instance.

## Prerequisites

1. A `qhana-plugin-runner` process serving the state-analysis plugins
   plus the `filesMaker` helper plugin and a circuit executor (e.g.
   `qiskit-simulator`).

   ```bash
   # From the repository root, in two terminals:
   poetry run flask run                     # API on :5005
   poetry run invoke worker                 # Celery worker
   ```

2. `PYTHONPATH` set so the `common` package is importable:

   ```bash
   export PYTHONPATH=$(pwd)/stable_plugins/state_analysis
   ```

3. By default the tests target `http://localhost:5005`. Override with the
   `QHANA_PLUGIN_RUNNER_URL` environment variable when the runner is on a
   different host or port:

   ```bash
   export QHANA_PLUGIN_RUNNER_URL=http://localhost:8080
   ```

## Running

From the repository root:

```bash
poetry run pytest stable_plugins/state_analysis/test/integration_test/ -v
```

Or a single test file:

```bash
poetry run pytest \
  stable_plugins/state_analysis/test/integration_test/test_lin_dep_api_qasm.py -v
```
