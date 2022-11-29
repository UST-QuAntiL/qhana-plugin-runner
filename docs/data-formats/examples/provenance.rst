Provenance (``provenance/*``)
=============================

+-----------------+--------------------------------------------------------------+
| data type       | provenance/execution-options, provenance/trace               |
+-----------------+--------------------------------------------------------------+
| content types   | application/json, (text/csv, application/X-lines+json)       |
+-----------------+--------------------------------------------------------------+

The ``provenance/*`` data type describes metadata related to the execution of executable artefacts.


Data Types
----------

provenance/execution-options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An entity defining execution options (e.g., the number of shots for quantum circuits or the cloud backend for classical deployment).
While the provenance execution options largely follow the data format for entities the format here is not confined to flat attribute values.
Provenance entities can have nested data structures.
Entities with nested data structures **must** be serialized as json!


provenance/trace
^^^^^^^^^^^^^^^^^^

An execution trace containing metadata about a single execution of an executable artifact.


Content Types
-------------

The same content types as for entities are supported, with the mayor difference being that nested values are possible and provenance data will often only be encountered in singular form.


Known Attributes
----------------

.. list-table::
    :widths: 20 20 60
    :header-rows: 1

    * * Attribute
      * Example
      * Description
    * * qpuVendor
      * ``IBM``
      * The vendor of the QPU (e.g., IBM, Google, Xanadu, etc.)

