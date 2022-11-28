# Add a recipe data type

* Status: [proposed]
* Deciders: []
* Date: [2022-10-05]

## Context and Problem Statement

There is a need to describe the interaction with a plugin such that it can be automated to some degree.
This includes providing defaults for some or all parameters.

## Use Cases

* PCA is computed once, but new points may have to be fitted to the same PCA instance
* Recommendations for plugin usage
* Human driven workflows

## Considered Options

* don't specify a special format
* use BPMN
* create a recipe format

## Decision Outcome

Chosen option: "create a recipe format", because ...


## Pros and Cons of the Options <!-- optional -->

### don't specify a special format

Do nothing because the status quo is already ok.

* Good, because nothing needs to be done
* Bad, because PCA (and similar plugins) still need a (custom) format for describing how new points are to be fitted

### use BPMN

Use an existing workflow language such as BPMN

* Good, because it is a full workflow definition
* Bad, because working with BPMN files is hard (if all constructs are to be supported)
* Bad, because this relies on a workflow engine

### create a recipe format

Create a custom recipe format.

* Good, because the format only needs to support the required features
* Good, because can be made easy to work with
* Good, because there is alwys the option to write a converter that outputs BPMN later
* Bad, because it is yet another workflow like language
* Bad, because it would require more code to make it work


<!-- markdownlint-disable-file MD013 -->
