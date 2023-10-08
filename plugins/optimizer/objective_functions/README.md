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
5. Update the parameters saved in the database in the **OptimizerCallbackProcess** class in the `routes.py` file.
6. Update the calculation of the number of weights needed for the objective function in the **PassDataEndpoint** class in the `routes.py` file.
7. Update the objective function you want to use in the **CalcCallbackEndpoint** class in the `routes.py` file.
8. Update the **CalcGradientEndpoint** and **CalcLossandGradEndpoint** functions in the `routes.py` file if you want to use them.
