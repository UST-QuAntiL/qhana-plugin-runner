# Data inputs in substeps

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2024-03-07]

## Context and Problem Statement

Data inputs are listed in the plugin metadata, but there is currently no way to determine if the data input is for the initial step or for any of the substeps of a multi step plugin.

## Decision Drivers <!-- optional -->

* Knowing the required data inputs before executing the plugin allows for checking if the data is available
* The data inputs contain information for automating API calls (i.e. which field should receive the data input)

## Considered Options

* Extend the file input metadata to include information about in which step the input is required

## Decision Outcome

Chosen option: "Extend the file input metadata", because this only requires minimal changes to the existing metadata.


## Pros and Cons of the Options <!-- optional -->

### Extend the file input metadata to include information about in which step the input is required

Add two new optional fields to the `dataInput` entries of the plugin metadata:

 *  `stepId` The id of the step that requires this input.\
    The input can still be optional for that step if the `required` field is `false`.
 *  `stepIsOptional` A boolean that if `true` signals that the step this input is required in may not appear during the plugin execution.
    If the step is not hit in the first place, then the input is also considered optional.


<!-- markdownlint-disable-file MD013 -->
