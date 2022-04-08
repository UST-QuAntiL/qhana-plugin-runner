# Specify Additional Endpoitns to used by Other Plugins

* Status: [proposed]
* Deciders: [Fabian Bühler]
* Date: [2022-04-08]

## Context and Problem Statement

The current plugin metadata and steps only allow plugins to specify a single URL as the API entry point.
While this also applies to the micro frontend, it can use multiple endpoints of the plugin.
If other plugins want to use the same functionality, they either need to use the micro frontend or guess the other endpoint URLs.

## Decision Drivers

* Plugins with the same functionality should be able to replace each other
* Plugin to plugin interaction should be possible without scraping a micro frontend

## Considered Options

* Use hardcoded relative URLs
* **Allow plugins to specify extra entry point URLs** (preferred option)


## Pros and Cons of the Options

### Use Hardcoded Relative URLs

Plugins could assume the existence of special URLs that are located relative to the specified entry point.
The existence of such URLs can be communicated by using special tags.

* Good, because no documentation change needed
* Bad, because the URLs must follow special (and arbitrary) rules and the endpoint location is limited by these rules

### Allow Plugins to Specify Extra Entry Point URLs

Allow plugins to specify extra URLs in their metadata and in steps.

* Good, because no hardcoded URLs and no special rules
* Good, because it allows the discovery of the special URLs by developers interacting with the plugin directly
* Bad, because it requires changes to the documentation and to other QHAna components

<!-- markdownlint-disable-file MD013 -->
