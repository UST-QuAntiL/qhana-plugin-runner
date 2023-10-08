# Minimizer Plugin

This folder contains the code for minimizer plugins.

## How a minimizer plugin works

* Each minimizer plugin provides a microfrontend where the user can set the parameters of the minimization.
* The processing endpoint of the microfrontend saves the parameters to the database and makes a callback to the provided callback URL.
* The minimization endpoint of the plugin is called by the coordinator plugin and it starts an asynchronous minimization task.

## How to create a minimizer plugin

Here is a step-by-step guide on how to create a minimizer plugin.
Nothing has to be implemented from scratch instead you can use an already existing minimizer plugin as a template.

1. Create a new folder in the `plugins/optimizer/minimizer` folder.
2. Copy the `__init__.py`, `schemas.py`, `tasks.py` files from an already existing minimizer plugin to the new folder.
3. Refactor the **_plugin_name**, **version**, **description**, **SecurityBlueprint** variable name and **class name** in the `__init__.py` file.
4. Update the **MinimizerSetupTaskInputSchema** in the `schemas.py` file to reflect the parameters of the minimization.
5. Update the parameters saved in the database in the **MinimizerSetupProcessStep** class in the `routes.py` file.
6. Update the minimization algorithm you want to use in the **minimize_task** function in the `tasks.py` file.
