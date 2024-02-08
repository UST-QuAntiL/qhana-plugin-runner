# Objective Function Plugins

A plugin that can evaluate an objective function used inside an optimization context.


Required Tags:

| Tag                   | Description |
|-----------------------|-------------|
| `objective-function`  | The identifying tag for plugins implementing this interface.     |

Optional Tags:

| Tag               | Description |
|:------------------|:------------|
| `gradient`        | If present, the plugin can also calcualte the gradient of the objective function, which is exposed through additional task specific links.      |


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

Objective function plugins are not required to have any outputs.


### Substeps

#### 1. optional plugin-specific steps

The plugin can expose additional steps for initializing parameters etc.

#### 2. `"pass_data"` **required**

After the initial interaction the plugin must expose a step with id `pass_data` to request the initial data for the calculation of the objective function:

 *  **Features** (parameter: `features`):\
    The data input **must** have the data-type [`entity/vector`](../data-formats/examples/entities.rst#entity-vector).

    Each entity is considered as 1 sample with k numeric features.
 *  **Target** (parameter `target`):\
    The data input **must** have the data-type [`entity/vector`](../data-formats/examples/entities.rst#entity-vector).

    Each entity is considered as 1 sample with exactly one numeric entry in the vector representing the target value.
    Each entity is only permitted exactly one target value!

:::{note}
Both inputs must have the same number of entities in the same order.
This also means that the ID columns of both inputs should be exactly the same.
:::

#### 3. optional plugin-specific steps

The plugin can expose additional steps for initializing parameters etc.

#### 4. `"evaluate"` **required**

Once the objective function is fully initialized, the plugin must expose a step with id `evaluate`.
During this step, the plugin **must** expose a task specific link with the type `of-evaluate` to evaluate the objective function.
The link is only active as long as the `evaluate` step is not cleared!

The evaluate step takes no inputs and can be completed simply by calling the corresponding endpoint.


### Interaction Endpoints

None


### Task Specific Interaction Endpoints

:::{warning}
All task specific calculation endpoints **can be async**!

Async endpoints will initially return a redirect to the resource that will later present the result.
The result resource must answer with HTTP status code `404` (not found) or `204` (no content) when the result is not yet ready.
When the result is ready, the resource must answer with HTTP status code `200` and the result.
:::


#### `of-weights` (active during step 4)

* Method: `get`

The endpoint is used to communicate the number of weights used by the objective function.

**Outputs:**

The output is a single JSON object with the key `weights` for the number of weights this objective function expects.


```json
{
    "weights": 11
}
```


#### `of-evaluate` (active during step 4)

* Method: `post`

The endpoint is used to evaluate the objective function.

**Inputs:**

The input for the calulation is the current weight vector as a JSON array inside a JSON object using the key `weights`.

```json
{
    "weights": [0.1, 0.8, 0, 0, 0.9]
}
```

**Outputs:**

The output of the loss calculation is a single JSON object with the key `loss` containing the result of the calculation (a single numeric value).

```json
{
    "loss": 0.1
}
```

#### `of-evaluate-gradient` (optional; active during step 4)

* Method: `post`

The endpoint is used to evaluate the gradient of the objective function.

:::{note}
An objective function plugin with the `gradient` tag **must** implement this task specific interaction endpoint!
:::

**Inputs:** same inputs as the `of-evaluate` endpoint.

**Outputs:**

The output is a single JSON object with the key `gradient` containing the gradient matrix as a list of lists.

:::{note}
Generally the `ndarray.tolist()` function does the right thing.
:::

:::{todo}
Specify which dimensions correspond to which axis of the matrix to avoid confusions.
:::

:::{todo}
If a gradient contains complex numbers, then this interface must be extended to also support the serialization of complex numbers.
Similar for very large gradients or weigth vectors!
:::

```json
{
    "gradient": [[0.1, 0.4], [0, 0.9]]
}
```

#### `of-evaluate-combined` (optional; active during step 4)

* Method: `post`

The endpoint is used to evaluate the objective function and its gradient in a combined step.

:::{note}
An objective function plugin with the `gradient` tag may implement this task specific interaction endpoint to allow for performance optimizations.
:::

**Inputs:** same inputs as the `of-evaluate` endpoint.

**Outputs:** a merged JSON object containing the keys of the `of-evaluate` and `of-evaluate-gradient` endpoints' outputs.

```json
{
    "loss": 0.1,
    "gradient": [[0.1, 0.4], [0, 0.9]]
}
```
