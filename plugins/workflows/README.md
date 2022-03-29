# Workflow plugin for BPMN and Camunda
Plugin: workflows@v0.2.0 ([changelog](#changelog))


## Prerequisites

Celery beat is required in order to run workflow instances.

```bash
# Run Celery beat
poetry run invoke beat
```

Additionally, [Camunda](https://camunda.com/) BPMN run is needed.

Celery workers completing workflow tasks should use a non-solo pool, e.g., gevent or eventlet.
Set the concurrency flag to allow multiple instances of a celery task.
Example:

```bash
# Run worker
poetry run invoke worker --pool=gevent  --concurrency=10
```

Currently, the python package `dataclasses-json` is a requirement for this plugin.
The package can be installed using pip:

```bash
# Install dataclasses-json
pip install dataclasses-json
```

## Creating workflows

TODO

## Changelog

### v0.2.0

- Use Celery tasks instead of threads
- Added invoke beat command to poetry
