# Use MICRO-Frontends to Expose Algorithm Parameters

* Status: [accepted]
* Deciders: [Fabian Bühler, Lukas Harzenetter]
* Date: [2021-06-29]

## Context and Problem Statement

Algorithms may have any number of parameters and hyper-parameters besides their main file inputs.
These parameters should be exposed to the user of the plugins.

## Decision Drivers

* All parameters and hyper-parameters should be accessible in QHAna
* Algorithms have a vide variety of parameters (in form and numbers)
* Parameters are also useful in other contexts (workflows, PlanQK)

## Considered Options

* define parameters in metadata and autogenerate the user interface
* **micro-frontend to expose parameters**
* use openapi specification and an expert interface

## Decision Outcome

Chosen option: "micro-frontend to expose parameters", because it allows plugins to define arbitrary parameters with great user experience while being easy to implement in QHAna.

### Positive Consequences

* QHAna can just embed micro frontends for the paramaters
* Parameters can be anything that can be captured in a HTML form
* Server-Side ui generation with opt-out for more complex micro frontends is possible
* Plugins have more control over the parameter ui than with a complex parameter definition language

### Negative Consequences

* All plugins must provide a micro frontend for QHAna
* Micro frontends can vary greatly in user experience

## Links

* <https://martinfowler.com/articles/micro-frontends.html>
* <https://micro-frontends.org>

<!-- markdownlint-disable-file MD013 -->
