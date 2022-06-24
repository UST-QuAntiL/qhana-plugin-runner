# Plugin Interaction Example

This folder contains two plugins: `interaction-demo` and `invokable-demo`.
`interaction-demo` is a multistep plugin that invokes the `invokable-demo` plugin during the first sub-step.
The micro frontend of the `invokable-demo` plugin is displayed in the first sub-step and the user can input data.

The flow of using the plugins is as follows:
1. user selects the `interaction-demo` plugin
2. first micro frontend of the `interaction-demo` plugin is displayed
3. user enters text and submits
4. micro frontend of the `invokable-demo` plugin is displayed as the first sub-step
5. user enters text and submits
6. second micro frontend of the `interaction-demo` plugin is displayed as the second sub-step
7. user enters text and submits

The processing steps of the two plugins share data with each other via a ProcessingTask object that is used to store and fetch data from the database.
