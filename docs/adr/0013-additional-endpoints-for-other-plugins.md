# Specify Additional Endpoints to used by Other Plugins

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2024-02-16]

## Context and Problem Statement

The current plugin metadata and steps only allow plugins to specify a single URL as the API entry point.
While this also applies to the micro frontend, it can use multiple endpoints of the plugin.
If other plugins want to use the same functionality, they either need to use the micro frontend or guess the other endpoint URLs.

## Decision Drivers

* Plugins with the same functionality should be able to replace each other
* Plugin to plugin interaction should be possible without scraping a micro frontend

## Considered Options

* Use hardcoded relative URLs
* **Allow plugins to specify extra entry point URLs**

## Decision Outcome

Chosen option: "Allow plugins to specify extra entry point URLs", because it is easy to implement and allows very complex interaction patterns with plugins.


## Pros and Cons of the Options

### Use Hardcoded Relative URLs

Plugins could assume the existence of special URLs that are located relative to the specified entry point.
The existence of such URLs can be communicated by using special tags.

* Good, because no documentation change needed
* Bad, because the URLs must follow special (and arbitrary) rules and the endpoint location is limited by these rules

### Allow Plugins to Specify Extra Entry Point URLs

Allow plugins to specify extra URLs in their metadata and in the task.

* Good, because no hardcoded URLs and no special rules
* Good, because it allows the discovery of the special URLs by developers interacting with the plugin directly
* Bad, because it requires changes to the documentation and to other QHAna components

#### Links in Metadata

Links in the plugin metadata get a new top level property `links` containinga list of links.
These links have two required properties (`href` and `type`).
The type must be unique among all links in the list and can be used to select a specific link from the list.

The semantic of a link (inputs, output and HTTP-method) should only depend on the link type.

Example:

```json
{
    "title": "Example Plugin",
    "description": "Example plugin exposing one link.",
    "links": [
        {
            "type": "greeting",
            "href": ".../example-plugin@v1/greeting/"
        }
    ]
}
```

#### Links in Tasks

For stateful interactions, links may need to be tied to a specific task and in some cases may only be valid for a limited amount of time.
These links should be presented as part of the task result in a similar top level `links` property.
All other properties of links are the same as for links in the plugin metadata.


<!-- markdownlint-disable-file MD013 -->
