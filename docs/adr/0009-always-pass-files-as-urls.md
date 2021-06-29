# Always Pass Files as URLs

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2021-06-29]

## Context and Problem Statement

QHAna (and workflows) save the outputs of a plugin to be used as inputs for another plugin.
The files (or their content) has to be passed to and from plugins.

## Decision Drivers

* results from plugins can be large or small
* results must be saved to allow retries of experiments, reproducible experiments etc. in QHAna

## Considered Options

* Let the plugin decide
* Always pass by value (file content)
* Give access to shared (cloud) storage to plugins
* **Always pass files by URL**

## Decision Outcome

Chosen option: "Always pass files by URL", because it normalizes file handling while allowing for many different storage options (e.g. local file system `file://`; cloud file system `https://`; passing file content directly `data://`).

### Positive Consequences

* Only one way to pass files to and from plugins reduces development effort

### Negative Consequences

* plugins must handle at least `file://` and `https://` URLs
* file URLs cannot require authentication (or authentication tokens must be passed to the plugin in some way)

<!-- markdownlint-disable-file MD013 -->
