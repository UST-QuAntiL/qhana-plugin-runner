# Subscribing for plugin updates

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2024-02-16]

## Context and Problem Statement

A plugin that invokes another plugin may encounter a situation where the invoked plugin unexpectedly asks for new user inputs in a step.
The invoking plugin needs a way to pass this step to the user while still being able to monitor the plugin result for changes.

## Decision Drivers <!-- optional -->

* Checking for result updates requires expensive polling.
* A normal plugin can suspend all execution until the user input is available saving resources
* Steps asking for new user input cannot be entirely predicted

## Considered Options

* Proxy the micro frontend
* Proxy the REST endpoint
* Proxy both
* **Provide a subscription mechanism for update events of the plugin result** (accepted)

## Decision Outcome

Implement a subscription mechanism for receiving task result updates via webhook notifications.


## Pros and Cons of the Options

### Proxy the micro frontend

Create a proxy frontend that displays the actual frontend for user input but listens to all messages between the proxied micro frontend and the application.
The proxy frontend can then notify the plugin when the user input was sent to the invoked plugin.
The invoking plugin would copy the step from the invoked plugin and replace the micro frontend url with the proxy frontend url.

* Good, because requires only a simple proxy micro frontend and a notification endpoint in the plugin
* Bad, because requires nesting iframes (only one level so this would be acceptable)
* Bad, because there is no way to detect direct calls to the REST endpoint

### Proxy the REST endpoint

Create a proxy endpoint that forwards the user input to the actual REST endpoint while notifying the plugin of the new input.
The invoking plugin would copy the step from the invoked plugin and replace the endpoint url with the proxy url.

* Good, because the proxy has complete access to the data being sent to the invoked plugin
* Good, because this catches direct api calls (to the proxy)
* Bad, because the micro frontend is most likely not aware of the new endpoint URL (the endpoint URL is often directly included in the micro frontend code)

### Proxy both

This is a combination of the first two proposals.
Both the micro frontend and the REST endpoint are proxied.

* Good, because (see first two options)
* Bad, because it requires more effort to implement two proxies

### Provide a subscription mechanism for update events of the plugin result

Require plugins (or the plugin runner) to implement a subscription mechanism that allows servers to subscribe to updates of plugin results.
The notification happens via webhook (e.g., by a post request to a specific url).

* Good, because this removes the need for polling entirely (for servers with webhooks)
* Bad, because it requires more effort to implement (only once in the plugin runner)
* Bad, because the subscription process inctroduces more complex behaviour

#### Subscribing to Events

A task result with a link of the type `subscribe` allows for registering webhooks to receive update events of that task.
Once a webhook subscribed to a specifc or all update events it will receive a `post` call with the url of the task result that is the `source` of the event and the event type.

#### Calling a Plugin and Subscribe

Since a task result is not available for registering a webhook before the entry point of a plugin was used to create the initial task result, there needs to be another mechanism in these cases.
For uncooperative plugins proxying the micro frontend and the processing endpoint is the only solution that can be guaranteed to work.
However, cooperative plugins can accept an extra parameter in both endpoints containing the url of the webhook to subscribe to events.
For the micro frontend this parameter must be passed as a query parameter, as this is the only reliable way to preserve/encode the parameter in the micro frontend link.
Such cooperative plugins can be identified by certain tags.

<!-- markdownlint-disable-file MD013 -->
