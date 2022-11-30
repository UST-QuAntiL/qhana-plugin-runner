Entities (``entity/*``)
=======================

+-----------------+--------------------------------------------------------------+
| data type       | entity/*                                                     |
+-----------------+--------------------------------------------------------------+
| content types   | text/csv, application/json, application/X-lines+json         |
+-----------------+--------------------------------------------------------------+

The ``entity/*`` data type describes the most generic entity format.
See :ref:`data-formats/data-model:entities` for more details.


.. note:: The examples in this document can (and should) be replaced with shortened real world examples once they are available to make testing new plugins easier.

Data Types
----------

entity/list
^^^^^^^^^^^

The data contains a list of entities.


entity/stream
^^^^^^^^^^^^^

The data contains a streamable list of entities that can be consumed line by line.

Allowed serialization formats for this type are: ``text/csv`` and ``application/X-lines+json``.

Plugins may treat ``application/json`` as ``application/X-lines+json`` if this data type is set.
However, they must fall back to processing the file in a non streaming manner if that fails.


entity/numeric
^^^^^^^^^^^^^^

Aside from the entity ``ID`` and ``href`` attributes every other attribute must be numeric (or a list of numbers).

Example:

.. code-block:: text

    ID,x,y,z
    entA,1,0.7,5
    entB,0.5,1,3


entity/vector
^^^^^^^^^^^^^

Stronger than ``numeric``, as every attribute aside from ``ID`` and ``href`` must be a single number.
The dimensions must be ordered lexicographically if order is important and the serialization format may not preserve attribute order (e.g. JSON).

Example:

.. code-block:: text

    ID,x,y,z
    entA,1,0.7,5
    entB,0.5,1,3


entity/matrix
^^^^^^^^^^^^^

Same as ``numeric``, every attribute aside from ``ID`` and ``href`` must be a single number (or a list of numbers).
Additionally, every attribute aside from ``ID`` and ``href`` must be an entity id.
Indexing the matrix should be done row first, meaning that the first index is for the row and the second for the column.

Example:

.. code-block:: text

    ID,entA,entB
    entA,1,0.7
    entB,0.5,1

.. code-block:: python

    matrix["entA"]          # Entity(ID="entA", entA=1, entB=0.7)
    matrix["entA"]["entB"]  # 0.7
    matrix["entB"]["entA"]  # 0.5


entity/attribute-metadata
^^^^^^^^^^^^^^^^^^^^^^^^^

The entities should be interpreted as attribute metadata entities describing properties of attributes of other entities.



Content Types
-------------

Entities ``text/csv``
^^^^^^^^^^^^^^^^^^^^^

Download: :download:`entities.csv <example_files/entities.csv>` 

.. code-block:: text

    ID,href,color
    paintA,example.com/paints/paintA,#8a2be2
    paintB,example.com/paints/paintA,#e9322d



Entities ``application/json``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download: :download:`entities.json <example_files/entities.json>` 

.. code-block:: json

    [
        {"ID": "paintA","href": "example.com/paints/paintA","color": "#8a2be2"},
        {"ID": "paintB","href": "example.com/paints/paintB","color": "#e9322d"}
    ]


Entities ``application/X-lines+json``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download: :download:`entities-lines.json <example_files/entities-lines.json>` 

.. code-block:: json

    {"ID": "paintA","href": "example.com/paints/paintA","color": "#8a2be2"}
    {"ID": "paintB","href": "example.com/paints/paintB","color": "#e9322d"}



