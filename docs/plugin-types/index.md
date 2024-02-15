# Special Plugin-Types

The main plugin types are covered by [](../plugins.rst). 
This document describes groups of plugins that implement a common interface and can be identified by the tags they use.


| Interface                  | Tags                                 | Description    |
|:---------------------------|:-------------------------------------|:---------------|
| [](circuit-executor.md)    | `circuit-executor`, [`qc-simulator`] | A plugin that is capable of executing quantum circuits. |
| [](minimizer.md)           | `minimizer`, [`gradient`]            | A plugin that can minimize objective functions. |
| [](objective-function.md)  | `objective-function`, [`gradient`]   | A plugin that computes the value of an objective function used in an optimization process. |


:::{toctree}
:glob:

*
:::


