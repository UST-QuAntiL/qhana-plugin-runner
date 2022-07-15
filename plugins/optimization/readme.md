# Optimizer + objective function plugins

## Optimizer plugin

Uses a gradient-free optimizer to optimize a given objective function.
This objective function is defined by an objective function plugin.

## Objective function plugin

Gets input data, hyperparameters and parameters.
Returns a single objective value that will get minimized by the optimizer plugin.

## Flow

1. User selects the optimizer plugin in the QHAna UI
2. User selects the objective function plugin in the micro frontend and submits
   1. The optimizer plugin adds a new step that will display the micro frontend of the selected objective function plugin
      1. The micro frontend gets a callback URL via a query parameter and enters it into the form 
3. User enters the hyperparameters into the micro frontend of the objective function plugin and submits
   1. The hyperparameters get stored in a file 
   2. The objective function plugin calls the callback URL to signal the optimizer plugin that the setup is complete and to send the number of parameters and the database ID that is required later
   3. The optimizer plugin adds a new step that will ask the user to select a dataset
4. User selects a dataset and submits
   1. The optimization task starts
      1. It repeatedly calls the calculation endpoint of the objective function plugin with the dataset URL, database ID and the current parameters
         1. The objective function plugin fetches the dataset, the hyperparameters (from the database which requires the database ID)
         2. The objective function calculates the objective values and returns it
