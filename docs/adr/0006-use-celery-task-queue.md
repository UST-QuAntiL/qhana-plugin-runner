# Use Celery Task Queue

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2021-06-29]

## Context and Problem Statement

Plugins can provide algorithms that would timeout any HTTP request.
They must have a way to execute long running tasks.

## Decision Drivers

* Most quantum computers are only accessible over a job queue
* QHAna plugins potentially get very large inputs

## Considered Options

* Ignore the potential for timeouts
* Let the plugin decide
* **Provide a configured task queue** (e.g. [Celery](https://docs.celeryproject.org/en/stable/index.html))

## Decision Outcome

Chosen option: "Provide a configured task queue", because it allows plugins to easily make use of the task queue for long running tasks.

### Positive Consequences

* Plugins can use Celery for simple and complex long running tasks

### Negative Consequences

* Plugin runner gets more complex and must handle task results/task state

<!-- markdownlint-disable-file MD013 -->
