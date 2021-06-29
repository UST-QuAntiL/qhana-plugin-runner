# Use Flask Template

* Status: [accepted]
* Deciders: [Fabian Bühler]
* Date: [2021-06-29]

## Context and Problem Statement

The REST API for the plugins requires a REST framework.

## Decision Drivers

* Flask is already known

## Considered Options

* [Flask](https://flask.palletsprojects.com/en/2.0.x/) without template
* [**Flask Template**](https://github.com/buehlefs/flask-template/)
* Other Framework

## Decision Outcome

Chosen option: "Flask Template", because it already provides boilerplate code with sane defaults already setup.

### Positive Consequences

* Less work to do
* Framework already known
* Can use Flask [Blueprints](https://flask.palletsprojects.com/en/2.0.x/blueprints/) to define APIs in a plugin

### Negative Consequences

* Template needs to be adapted slightly
