# Schema links

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2024-03-07]

## Context and Problem Statement

Plugins can communicate their data inputs via their metadata. However, any other input currently relies on either the user correctly reading the microfrontend, special restrictions for certain plugin types (i.e. circuit executor plugins), or out of band information in the form of the OpenAPI specification served by the plugin runner.

How can we provide information about non data inputs in the plugin metadata?

## Decision Drivers <!-- optional -->

* Do not break compatibility too much
* Allow for automation

## Considered Options

* Add a third link keyword for links to schemas

## Decision Outcome

Chosen option: "Add a third link keyword for links to schemas", because it only requires small changes to the metadata and is entirely optional.


## Pros and Cons of the Options

### Add a third link keyword for links to schemas

Currently the plugin metadata and results can contain links using `href` for the REST endpoint and `uiHref` for micro frontends.
In all places where these fields are used, we can add an optional third field `schema` to link to a schema that describes the inputs for the REST endpoint.

* Good, because it requires only minimal additions to the current metadata
* Good, because it allows for automatic discoverability of the inputs for REST endpoints without using the micro frontend
* Bad, because it may require changes in tools consuming the plugin metadata
* Bad, because the usefulness may not be immediate

<!-- markdownlint-disable-file MD013 -->
