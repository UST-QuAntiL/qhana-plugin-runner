# Plugins Must Provide Metadata

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2021-06-29]

## Context and Problem Statement

In order for QHAna to use the plugins and present them to the user in a meaningful manner the plugins must provide some form of metadata describing themselves.

## Decision Drivers

* QHAna must present the plugins to the user (ideally only relevant plugins for the current experiment state)

## Considered Options

* No Metadata
* Unstructured Metadata
* **Structured Metadata**

## Decision Outcome

Chosen option: "Structured Metadata", because this allows for some decisions (e.g. plugin grouping or checking if plugin can be used with certain inputs) to be automatically checked greatly enhancing the user experience.

The exact metadata format is to be defined in the documentation.

### Positive Consequences <!-- optional -->

* Some decisions can be automated
* Better user experience
* Not much effort for plugins (compared to unstructured metadata)

### Negative Consequences <!-- optional -->

* Plugins must provide valid metadata to work correctly with QHAna

<!-- markdownlint-disable-file MD013 -->
