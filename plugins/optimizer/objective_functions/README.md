# Objective Function Plugin

This folder contains the code for objective function plugins.

## How the objective function plugin works

* Each objective function plugin provides a microfrontend where the user can set the parameters of the objective function.
* The processing endpoint of the microfrontend saves the parameters to the database and makes a callback to the provided callback URL.
* The `pass_data`` endpoint retrieves the input and target data and returns the number of weights needed for the objective function.
* The `calc_loss` endpoint calculates the loss of the given weights and returns it.
* The `calc_gradient` endpoint calculates the gradient of the given weights and returns it (optional).
* The `calc_loss_and_gradient` endpoint calculates the loss and gradient of the given weights and returns them (optional).
* A caching mechanism is used by the ridge-loss plugin to speed up the calculation of the loss.

## How to create an objective function plugin

Here is a step-by-step guide on how to create an objective function plugin.
Nothing has to be implemented from scratch instead you can use an already existing objective function plugin as a template.

1. Create a new folder in the `plugins/optimizer/objective_functions` folder.
2. Copy the `__init__.py`, `schemas.py`, `tasks.py` files from an already existing objective function plugin to the new folder.
3. Refactor the **_plugin_name**, **version**, **description**, **SecurityBlueprint** variable name and **class name** in the `__init__.py` file.
4. Update the **HyperparamterInputSchema** in the `schemas.py` file to reflect the parameters of the objective function.
For the *hinge-loss* plugin the **HyperparamterInputSchema** takes c as a parameter and looks like this:

```python
class HyperparamterInputSchema(FrontendFormBaseSchema):
    c = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Regularization parameter",
            "description": "Regularization parameter for Hinge Loss function.",
            "input_type": "textarea",
        },
    )
```

5. Update the parameters saved in the database in the **OptimizerCallbackProcess** class in the `routes.py` file.
The *hinge-loss* plugin saves the c parameter like this:

```python
db_task.data["c"] = arguments.c
```

6. Update the calculation of the number of weights needed for the objective function in the **PassDataEndpoint** class in the `routes.py` file.
The *hinge-loss* function needs the same number of weights as the number of features:

```python
return {"number_weights": input_data.x.shape[1]}

```

7. Update the objective function you want to use in the **CalcCallbackEndpoint** class in the `routes.py` file.
The *hinge-loss* function looks like this:

```python
    loss = hinge_loss(
        X=input_data.x,
        y=input_data.y,
        w=input_data.x0,
        C=db_task.data["c"],
    )
```

While the `hinge_loss` function is defined in the `tasks.py` file.

8. Update the **CalcGradientEndpoint** and **CalcLossandGradEndpoint** functions in the `routes.py` file if you want to use them.
This is optional and not needed for the *hinge-loss* plugin.
The implementation for those two functions is the same as for the **CalcCallbackEndpoint** function.
An example implementation can be found in the [neural-network plugin](./neural_network/routes.py).

## Use caching for the objective function

As the objective function calculation endpoint loads the input and target data and the hyperparameters from the database it can be slow.
To speed up the calculation of the objective function the *ridge-loss* plugin uses a caching mechanism.
Two changes are needed to use the caching mechanism:

1. Update the parameters saved in the database in the **OptimizerCallbackProcess** class in the `routes.py` file.
This time the parameters are saved in a dictionary:

```python
hyperparameter = {"alpha": arguments.alpha}
db_task.data["hyperparameter"] = hyperparameter
```

2. Update the objective function in the **CalcCallbackEndpoint** class in the `routes.py` file.
Instead of loading the data from the database use the `get_of_calc_data` function to load the data from the cache or the database:

```python
x, y, hyperparameter = get_of_calc_data(db_id)
```
