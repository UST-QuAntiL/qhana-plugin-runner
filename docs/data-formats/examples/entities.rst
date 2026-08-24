Entities (``entity/*``)
=======================

+-----------------+--------------------------------------------------------------+
| data type       | entity/*                                                     |
+-----------------+--------------------------------------------------------------+
| content types   | text/csv, application/json, application/X-lines+json         |
+-----------------+--------------------------------------------------------------+

The ``entity/*`` data type describes the most generic entity format.
See :ref:`data-formats/data-model:entities` for more details.


.. todo:: The examples in this document can (and should) be replaced with shortened real world examples once they are available to make testing new plugins easier.

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


entity/shaped_vector
^^^^^^^^^^^^^^^^^^^^

Similar as ``enitity/vector``, with the addition that each vector has a shape.
The dimensions may not start with ``shape`` and be ordered lexicographically if order is important and the serialization format may not preserve attribute order (e.g. JSON).
The shapes must start with ``shape`` and be ordered lexicographically if order is important and the serialization format may not preserve attribute order (e.g. JSON).

Example:

.. code-block:: text

    ID,shape0,shape1,dim0,dim1,dim3,dim4
    entA,2,2,0.5,1,0.7,5
    entB,2,2,3,0.5,1,3

.. code-block:: python
    shaped_vector["entA"]       # [[0.5, 1], [0.7, 5]]
    shaped_vector["entA"][0][0] # 0.5
    shaped_vector["entA"][0][1] # 1
    shaped_vector["entA"][1][0] # 0.7
    shaped_vector["entA"][1][1] # 5


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


entity/label
^^^^^^^^^^^^^

Each entity has only three attributes ``ID``, ``href`` and ``label``. The ``label`` attribute can be any type of object.

Example:

.. code-block:: text

    ID,label
    entA,"cat"
    entB,"dog"


entity/attribute-metadata
^^^^^^^^^^^^^^^^^^^^^^^^^

The entities should be interpreted as attribute metadata entities describing properties of attributes of other entities.

.. seealso:: :ref:`data-formats/data-loader-formats:attribute metadata`


entity/dimension-mapping
^^^^^^^^^^^^^^^^^^^^^^^^

Records where each dimension of an ``entity/vector`` file came from, so the link to the original input is preserved when plugins such as :ref:`vector-concat` merge several vector files and renumber their dimensions.
The entities are listed one per output dimension, ordered by dimension index, with the ``ID`` naming the output dimension the entity describes.

+--------------------+---------------------------------------------------------------------+
| attribute          | description                                                         |
+====================+=====================================================================+
| ``ID``             | name of the output dimension, e.g. ``dim5``                         |
+--------------------+---------------------------------------------------------------------+
| ``inputIndex``     | 0-based position of the input file in the plugin's input order      |
+--------------------+---------------------------------------------------------------------+
| ``source``         | name of the input file the dimension came from                      |
+--------------------+---------------------------------------------------------------------+
| ``sourceUrl``      | url of the input file (the url of the archive for zipped inputs)    |
+--------------------+---------------------------------------------------------------------+
| ``zipMember``      | name of the file inside the archive, empty for unzipped inputs      |
+--------------------+---------------------------------------------------------------------+
| ``sourceDimension``| name of the column in the input file                                |
+--------------------+---------------------------------------------------------------------+

.. code-block:: json

    [
        {"ID": "dim0", "href": "", "inputIndex": 0, "source": "color.json",
         "sourceUrl": "http://localhost:5005/files/17/download/vectors.zip",
         "zipMember": "color.json", "sourceDimension": "dim0"},
        {"ID": "dim1", "href": "", "inputIndex": 0, "source": "color.json",
         "sourceUrl": "http://localhost:5005/files/17/download/vectors.zip",
         "zipMember": "color.json", "sourceDimension": "dim1"},
        {"ID": "dim2", "href": "", "inputIndex": 1, "source": "shape.json",
         "sourceUrl": "http://localhost:5005/files/17/download/vectors.zip",
         "zipMember": "shape.json", "sourceDimension": "dim0"}
    ]

Produced by:

  * :ref:`vector-concat`


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



