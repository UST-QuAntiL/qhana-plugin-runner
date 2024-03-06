# Quantum Circuit Executor Plugins

A quantum circuit executor plugins can, as the name already implies, execute quantum circuits.

:::{note}
Currently, quantum circuit always refers to circuits for **gate based** quantum computers.
:::


Required Tags:

| Tag                   | Description |
|-----------------------|-------------|
| `circuit-executor`    | The identifying tag for plugins implementing this interface.     |

Optional Tags:

| Tag               | Description |
|:------------------|:------------|
| `qc-simulator`    | If present, the plugin uses a quantum computing simulator.      |
| `qiskit`          | A tag specifying the (main) SDK being used by the plugin.      |
| `qasm` `qasm-2`   | The language that this plugin can understand. The tag `qasm` (for OpenQASM) should always be accompanied by the specific version of qasm supported.     |



## Interface

```{contents}
:depth: 2
:local: True
```


### Main Interaction

The main input of the plugin is therefore such a circuit along with specific execution options.
Both must be provided in the starting step of the plugin flow, but the execution options are optional.
The plugins can make some execution options directly available as parameters for the user.
However, all these parameters must be optional so that the circuit executors can also be used in more automated settings.

Some circuit executor plugins may require authentication credentials in order to access the actual quantum execution resource.
Those plugins can expose the authentication form as part of the starting step as optional inputs.
If the credentials are not provided in the first step, the plugin should ask for the credentials again in a substep form.

:::{hint}
The circuit executor plugin **must** execute the circuit **as is**, without adding or removing any measurements.
Transpiling the circuit to an equivalent circuit is allowed.
:::


### Inputs

Circuit executor plugins can have two types of data (i.e. file based) inputs:

 *  **Quantum Circuit** (parameter: `circuit`):\
    The data input **must** have the data-type [`executable/circuit`](../data-formats/examples/executables.rst#executable-circuit).
 *  **Execution Options** (parameter `executionOptions`):\
    The data input **must** be available, but may not receive a value (i.e., this input should be treated as **optional**).
    The data input **must** have the data-type [`provenance/execution-options`](../data-formats/examples/provenance.rst#provenance-execution-options).

    The execution options **must** support `shots` as the number of shots to simulate/execute.
    Unknown execution options must be ignored.
    Execution options that are specific to specific plugins should have unique names.

    | Attribute | Description                                      |
    |:----------|:-------------------------------------------------|
    | `shots`   | The number of shots to run the circuit.          |


Circuit executor plugins can have two types of direct parameter inputs:

 *  **Execution option overrides**: circuit executor plugins can expose some of the execution options (usually received through the `executionOptions` parameter) as additional parameters.
    These parameters **must** all be **optional** inputs.
    If they are set, they override the values from the file based execution options.

    For example, the number of shots can be exposed as an additional parameter so that users can directly set this option from the plugin ui without first constructing the execution options data.
 *  **Authentication details**: this can include username+password or API tokens.
    The authentication parameters **must** be **optional**.

### Outputs

Circuit executor plugins must have the following required data outputs:

 *  **result-counts.json** / **result-counts.csv**\
    The counts **must** be of a numeric data type such as [`entity/numeric`](../data-formats/examples/entities.rst#entity-numeric) or [`entity/vector`](../data-formats/examples/entities.rst#entity-vector).

    Result counts should contain binary strings that follow the convention of qiskit.
    Only explicit measurements should be recognized for the result counts.
    For circuits defined in OpenQASM this means that only classical registers will be used for the result counts.
    A circuit without any measurements will present no counts.
    Use an empty binary string `""` with the number of shots in this case.
 *  **result-trace.json**\
    The result trace **must** have the data type [`provenance/trace`](../data-formats/examples/provenance.rst#provenance-trace).
    This output contains provenance data for that specific execution of a circuit.

    | Attribute | Description |
    |:----------|:------------|
    | TODO      | --          |
 *  **execution-options.json**\
    The execution options output **must** have the data-type [`provenance/execution-options`](../data-formats/examples/provenance.rst#provenance-execution-options).

    Circuit executors **must** output execution options with which the execution of the circuit can be reproduced as closely as possible.
    This means that they should explicitly add all default options to the execution options.
    For simulators (or for the transpiling step) which can use a fixed seed to create fully reproducible results, this seed should be included in the execution options.
    Options set by the user in the starting step should be output as is, with some exceptions:

    | Attribute         | Description |
    |:------------------|:------------|
    | `start-session`   | This flag should never be output in the execution options.           |
    | `api-token`       | Passwords, etc. should never be output in the execution options.     |
    | `username`        | Passwords, etc. should never be output in the execution options.     |
    | `password`        | Passwords, etc. should never be output in the execution options.     |


If available, circuit executors can optionally output different representations of the results:

 *  **result-statevector.json** / **result-statevector.csv**\
    The state vector **must** have the data type [`entity/vector`](../data-formats/examples/entities.rst#entity-vector).
    The number values are complex numbers as represented by python (using `j` instead of `i` for the imaginary part).

    The state vector describes the qubit states of **all** qubits after executing the circuit **including** any measurement.
    If all qubits are measured, the state vector will only include `0` or `1` amplitudes as the measurements have collapsed any superposition!
 *  **result-distribution.json** / **result-distribution.csv**\
    Same as result counts, but instead of counts the values are a probability distribution.

    All values must be $0 \leq x \leq 1$ and the sum of all values must be exactly $1$.


### Substeps

#### 1. `"authentication"` *(optional)*

A substep for requesting the user to provide authentication credentials for the quantum computer or simulator the plugin uses.
This substep should only be used if authentication is required and was not provided already.

The inputs to this step can be an API Token, a username+password pair or any other form of authentication data.
This step is always expected to be filled out by a user.


### Interaction Endpoints

#### `devices` *(optional)*

* Method: `get`

The endpoint is used to retrieve a list of devices that the circuit executor can directly access.

:::{note}
The list must only include devices that the circuit executor can use directly to compute the quantum circuit result.
:::

**Inputs:** the get input does not require any inputs, but an API token may be provided using the `Authorization` header.
If a token is available, then the plugin should return an up-to-date list of available backends.

**Outputs:** a JSON list containing an object for each device with information about that device.
The attributes `name` and `vendor` are required and together they form the unique device identidfier.
The optional attributes `title` and `description` can be used to include a human readable title and additional information about the device that should be displayed to a user.
Optionally, the attribute `available` can be used to signal which of the backends is currently available for execution.
Additionally, further information about the backend may be provided.

```json
[
    {
        "name": "<the device identifier name>",
        "vendor": "<the vendor of the device>",
        "title": "<Device title to present in UIs>",
        "description>": "<device description>",
        "available": false,
        "...": "..."
    },
    {
        "name": "ibm_sherbrooke",
        "vendor": "ibm",
        "title": "IBM Sherbrooke",
        "description>": "127 Qubits, Eagle r3",
        "available": false,
        "qubits": 127,
        "CLOPS": 5000
    }
]
```




