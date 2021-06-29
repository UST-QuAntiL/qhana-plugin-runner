# Use Python

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2021-06-29]

## Context and Problem Statement

The plugins need to be written in a suitable programming language.

## Decision Drivers

* The algorithms are already implemented as prototypes in python
* Many quantum computing SDKs use Python (same for machine learning)

## Considered Options

* Java
* **Python**

## Decision Outcome

Chosen option: "Python", because most of the plugins would have to be implemented (at least partially) in Python.

### Positive Consequences

* QHAny algorithm prototypes can be reused

### Negative Consequences

* Python knowledge is less common than Java knowledge

<!-- markdownlint-disable-file MD013 -->
