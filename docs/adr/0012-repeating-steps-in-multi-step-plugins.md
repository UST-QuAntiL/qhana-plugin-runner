# Repeating Steps in Multi-Step Plugins [undecided]

* Status: [proposed]
* Deciders: [Fabian Bühler]
* Date: [2022-04-08]

## Context and Problem Statement

Multi-step plugins can ask the user for input multiple times during a stateful computation.
However, each step is recorded in an ever growing step list in the pending result.
This is not suitable for algorithms that have an optimazation step where the same operation is performed in a loop until a breaking condition is satisfied.

## Decision Drivers

* All user inputs should be recorded by the QHAna backend
* Plugins may use other plugins with such reapeatable steps => they need a way to discern them as repeatable

## Considered Options

* Just record the last user input
* Allow plugins to omit previous cleared steps from the step list
* **Allow steps to be marked as repeatable** (preferred option)


## Pros and Cons of the Options <!-- optional -->

### Just Record the Last User Input

Plugins can handle such iterations without the knowledge of the host application by acting as if the step was not cleared until the last iteration occured.
For the application using this plugin, it will appear as a single step.

* Good, because no adjustments needed
* Bad, because there is no way to record all user inputs
* Bad, because plugins can implement this in many different ways

### Allow Plugins to Omit Previous Cleared Steps from the Step List

Allowing plugins to remove cleared steps again from the step list would keep the list short, even if a step is repeated often.

* Good, because only minimal changes to documentation / other components needed
* Good, because it would dramatically increase the possible number of steps aplugin can have
* Bad, because it would dramatically increase the possible number of steps aplugin can have
* Bad, because it is hard to make sure that every step is recorded correctly / it is hard to match the incomplete list against the already recorded steps

### Allow Steps to be Marked as Repeatable

Define how steps can be marked as repeatable (e.g. with a boolean flag) and how such repeatable steps can be cleared (in case they must be manually set as cleared).
Each iteration of such a repeatable step would semantically behave as a new step for which input can be recorded, but the step list only grows once the repeatable step is finally cleared.

* Good, because repeatable steps are distinguishable on every level (UI and API)
* Good, because repeatable steps can use a single step ID
* Bad, because it requires some changes in the documentation and other QHAna components


<!-- markdownlint-disable-file MD013 -->
