# Minimizer Plugins

A plugin that can evaluate an objective function used inside an optimization context.


Required Tags:

| Tag                   | Description |
|-----------------------|-------------|
| `minimizer`           | The identifying tag for plugins implementing this interface.     |

Optional Tags:

| Tag               | Description |
|:------------------|:------------|
| `gradient`        | If present, the plugin can use gradients some objective functions supply during minimization.      |


## Interface

```{contents}
:depth: 2
:local: True
```


### Main Interaction

Both, the micro frontend and the processing endpoint accept a `callback` paramater in the query string of the URL.
This parameter contains a webhook that will be automatically subscribed to all task updates (especially updates to task status and steps).

The main interaction is otherwise completely up to the objective function plugin.
However, the plugin must expose certain steps and task specific links in later parts of the interaction.


### Inputs

Initial inputs are up to the plugin.


### Outputs

The minimizer must output the final weights of the minimization process.

 *  **weights.json** / **weights.csv**\
    The weights **must** be of the type [`entity/vector`](../data-formats/examples/entities.rst#entity-vector).


### Substeps

#### 1. optional plugin-specific steps

The plugin can expose additional steps for initializing parameters etc.

#### 2. `"minimize"` **required**

After the initial interaction the plugin must expose a step with id `minimize`.

The step has 1 required input:

 *  **Objective Function** (parameter: `objectiveFunction`):\
    The objective function is provided in the form of a URL pointing to an objective function task that is in the [`evaluate` step](./objective-function.md#evaluate-required).

Additionally, the step has one optional input:

 *  **Initial Weight** (parameter `initialWeights`):\
    The data input **must** have the data-type [`entity/vector`](../data-formats/examples/entities.rst#entity-vector).

    A single entity with numeric data attached that represent the initial weights.
    All weights must be between 0 and 1!

    The plugin must always allow this input, but callers of this plugin may not always provide a value for this input.

#### 3. optional plugin-specific steps

The plugin can expose additional steps.


### Interaction Endpoints

None
