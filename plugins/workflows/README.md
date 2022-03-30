# Workflow plugin for BPMN and Camunda
Plugin: workflows@v0.4.0 ([changelog](#changelog))


## Prerequisites

Python >= 3.9

Celery beat is required in order to run workflow instances.

```bash
# Run Celery beat
poetry run invoke beat
```

Additionally, [Camunda](https://camunda.com/) BPMN run is needed.

Celery workers completing workflow tasks should use a non-solo pool, e.g., gevent or eventlet.
Set the concurrency flag to allow multiple instances of a celery task.
Example:

```bash
# Run worker
poetry run invoke worker --pool=gevent  --concurrency=10
```

Currently, the python package `dataclasses-json` is a requirement for this plugin.
The package can be installed using pip:

```bash
# Install dataclasses-json
pip install dataclasses-json
```

## Creating workflows

### Service Tasks

QHAna plugins are represented as Service Tasks in the modeler.
Note: You can choose the name of the service task freely.

![Example Service Task](./docs/service-task.png)

### Linking

In order for the workflow plugin to know which QHAna plugin should be called when a
token reaches the service task, set the Implementation option of the service task
to `External` and the topic to `plugin.name`, where `plugin.` is the prefix and `name`
the name of the plugin.

![Example Linking](./docs/linking.png)

### Plugin Output

It is possible to assign a variable to the output of a QHAna plugin. To do this,
add an entry to the Output Parameters option of the service task. In the Process
Variable Name field you can specify the variable name as `qoutput.name`, where
`qoutput.` is the prefix and `name` can be any name that is unused. The process
variable name is used to map output to input. 

The varibale assignment type should be set to `String or Expression` and the
variable assignment value field to `${output}`.

Note: The variable will contain all results of the plugin. The [Inputs](#inputs)
section discusses how to select a specific result as input.

![Example Output](./docs/output.png)

### Plugin Inputs

A QHAna plugin may have one or more inputs. To define inputs in the Camunda
modeler, add an entry to the Input Parameters option of the service task.
In case you want to use the output of a QHAna plugin as an input, set the 
Local Variable Name field to `qinput.name` where `qinput.` is
the prefix and `name` corresponds to the plugin input parameter name.

Additionally, you may use any variable from the BPMN workflow as input, therefore
the Local Variable Name should be set to `variable_name`, the name of the variable.

The Variable Assignment Type should be set to `Map`. To map an output to the input
add an entry. The `Key` should be set to the variable name of the output (See 
[Output](#output) section for details). There are different options to set
the `Value` field:

- `name: output.txt` - With `name` you can select the result by the name property 
from a QHAna output
- `dataType: wu-palmer-cache` - With `dataType` you can select the result by the data
type property from a QHAna output
- `plain` - Use this option if the input is not from the output of a QHAna 
plugin. (Variable name is not `qoutput.name`)

You can add multiple inputs to the Input Parameters option of the service task.

![Example Inputs](./docs/inputs.png)


### Workflow outputs

To mark a variable as workflow output add the prefix `return.`. 

Example:

- `qoutput.someVariable` to `return.qoutput.someVariable`
- `myVariable` to `return.myVariable`


### Human Tasks / Workflow Inputs

For a workflow inputs from Human Tasks may be required. It is
possible to create a form within the Camunda Modeler that will be displayed
in QHAna to receive the required inputs.

First, add a user task in the Camunda Modeler.

![Example Human Task](./docs/human-task.png)

Within the `Forms` Tab of the user task leave the `Type` to `<none>`. For every
required input add an entry to the `Form Fields` option. 

Set the `ID` to the variable name which will be used in the workflow. The `Type`
option should be set to `string`. 

The `Label` can be set to any value. The label is placed above the text box for the 
input in a QHAna form. `Default Value` is used for the text box for
the input. Both can be left empty, but it is recommended to use the label.

To display a select field within a QHAna form add the prefix `choice:` to the 
`Default Value` field.
Example: `choice: typeA, typeB, typeC`.

![Example Human Task Form](./docs/human-task-form.png)

### Exceptions

Service Tasks representing QHAna plugins may throw BPMN exceptions. For this
an error boundary event is added to the service task.

![Example Exceptions](./docs/exceptions.png)

Currently, following exception error codes are supported:

- qhana-plugin-failure: Thrown when a QHAna plugin fails
- qhana-unprocessable-entity-error: Thrown when the input of a QHAna plugin is
an unprocessable entity
- qhana-mode-error: Thrown when the input mode is not one specified in 
[Inputs](#inputs)

### Example Workflows

You can find examples in the `/bpmn` directory of the workflows plugin to get started.

## Changelog

### v0.4.0
- Support select form fields for QHAna forms

### v0.3.0
- Mark workflow variables as workflow output
- Plugin now terminates with a list of workflow output variables

### v0.2.0

- Use Celery tasks instead of threads
- Added invoke beat command to poetry
