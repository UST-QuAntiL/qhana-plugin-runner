# Plugin Runner as Seperate Project

<!-- Use number 3 for first actual ADR to be consistent with the numbering in the documentation. -->

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2021-06-29]

## Context and Problem Statement

The QHAna tool must be refactored (rewritten, even rearchitectured) to support plugins for new algorithms.
The plugins should be exposed (and consumed) via a REST API to benefit other projects (e.g. PlanQK).
Some boilerplate code useful for all plugins can and should be bundled into a library.

## Decision Drivers

* QHAna must support plugins
* Plugins should benefit many projects -> REST API

## Considered Options

* Python only plugins
* Python only plugins, but exposed as REST API by QHAna
* Shared library for boilerplate code
* **Seperate plugin runner project** (in this repostiory)

## Decision Outcome

Chosen option: "Seperate plugin runner project", because this provides the best compromise between easy plugin development (like "Python only plugins") while also allowing to deploy plugins individually for their REST API.

### Positive Consequences

* Most biolerplate code can be shared
* Config loading, etc. can be done by the plugin runner
* Plugins only need to implement their business function/their algorithm

### Negative Consequences

* Plugins must be implemented in python
* Plugins cannot define arbitrary APIs

## Pros and Cons of the Options

### Python only plugins

Plugins directly loaded by QHAna cannot be used by other projects.

### Python only plugins, but exposed as REST API by QHAna

If QHAna exposes a REST API for using the loaded plugins they can be used by other projects.

* Good, because plugins could use all QHAna functions
* Good, because plugins only need to implement a small interface while they are accessible via the REST API
* Bad, because individual deployments would need to deploy QHAna to teploy a single plugin

### Shared library for boilerplate code

* Good, because code can be shared and plugins can be smaller
* Bad, because many features require more than just importing a function and calling it (REST API, async tasks, config loading)

### Seperate plugin runner project

Inversion of control (runner controls plugin) compared to "Shared library for boilerplate code".

* Good, because runner can act as library for shared code
* Good, because runner can provide more functionality than a library (like "Python only plugins")
* Good, because runner has less overhead than the full QHAna tool
* Bad, because runner must be loaded for each deploy (but can load multiple plugins)


<!-- markdownlint-disable-file MD013 -->
