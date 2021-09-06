# Complex Plugin Interactions

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2021-09-03]

## Context and Problem Statement

Plugins can already create and modify data.
However, a plugin can only directly expose one processing resource to the QHAna UI.
This limits the possible interactions with a plugin.
Plugins cannot cooperate with other plugins or ask for user input in a multi step process.

## Decision Drivers

* We want to enable plugin cooperation (e.g. an optimizer plugin optimizing any optimizable quantum circuit)
* We want to enable multi step processes where the user may be required to input data at every step

## Considered Options

* Implement Workflows or integrate a Workflow Tool (like Camunda) + Engine
* Hardcoded multi-plugin templates
* Multi step micro frontends

## Decision Outcome

Chosen option: "Multi step micro frontends", because it is easy to implement and allows very complex interaction patterns with plugins.

### Positive Consequences

* Complex plugin interactions will be possible
* Complex interactions must not be hardcoded (like in option "Hardcoded multi-plugin templates")
* Allows creation of meta plugins that compose multiple plugins into one functionality
* Easy to implement

### Negative Consequences

* Meta plugins are not real workflows
* Creating more and more complex micro frontends is neccessary to take advantage of this feature


## Pros and Cons of the Options

### Implement Workflows or integrate a Workflow Tool (like Camunda) + Engine

Implement full workflow support in QHAna

* Good, because it would allow basically any interaction pattern with plugins
* Good, because workflows are powerful
* Bad, because workflows are not easy to create learn (steep learning curve, technical/deployment details need experts)
* Bad, because workflows are typically not executed one activity at a time (while in QHAna executing one Plugin at a time is the norm)
* Bad, because integrating workflows is a huge task and thus hard to implement

### Hardcoded multi-plugin templates

For typical multi plugin use cases one could implement hardcoded workflow like multi-plugin templates that specify how and in which order to invoke plugins.
The user would select the plugins for the template in a first step and then be guided through the whole process.
Plugins provide metadata (e.g. tags describing what type of plugin they are) that can be used for this.

* Good, because allows for some complex interaction patterns involving multiple plugins
* Good, because relatively easy to implement
* Bad, because the approach is not flexible
* Bad, because the approach needs to be implemented in the QHAna UI

### Multi step micro frontends

Instead of directly producing data a plugin could return a link to a new micro frontend.
This new micro frontend would then be presented to the user who could input the required data.
This can be repeated until a final result can be presented by the plugin.

* Good, because it allows for many complex interaction patterns with one or multiple plugins
* Good, because it is easy to implement in the frontend
* Bad, because it is harder to implement for plugins (but should not be neccessary for most plugins)

<!-- markdownlint-disable-file MD013 -->
