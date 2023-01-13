Executables (``executable/*``)
==============================

+-----------------+--------------------------------------------------------------+
| data type       | executable/*                                                 |
+-----------------+--------------------------------------------------------------+
| content types   | text/x-python, text/javascript, application/java-archive,    |
|                 | text/x-qasm                                                  |
+-----------------+--------------------------------------------------------------+

The ``executable/*`` data type describes executable code.


Data Types
----------

executable/[c|h|q]-function
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An executable containing a single classical, hybrid or quantum function.
A quantum function is just a partially realized quantum circuit.

.. todo:: Decide what to do if multiple functions are present. Options: first function, last function, decorated function, special function name

.. todo:: Specify how code can define runtime dependencies (e.g. with special comments)


executable/circuit
^^^^^^^^^^^^^^^^^^

The data is a full quantum circuit (that may be parameterized).


executable/workflow
^^^^^^^^^^^^^^^^^^^

Data contains an executable workflow (e.g. a BPMN2 workflow).


executable/bundle
^^^^^^^^^^^^^^^^^

A prebuilt binary or package that can be used by other executable artifacts.



Content Types
-------------

The content types depend on the programming language used.
