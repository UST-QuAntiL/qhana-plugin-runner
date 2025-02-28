Relations (``relation/*``)
==========================

+-----------------+--------------------------------------------------------------+
| data type       | relation/*                                                   |
+-----------------+--------------------------------------------------------------+
| content types   | text/csv, application/json, application/X-lines+json         |
+-----------------+--------------------------------------------------------------+

The ``relation/*`` data type describes the most generic relation format.
See :ref:`data-formats/data-model:relations` for more details.


.. todo:: The examples in this document can (and should) be replaced with shortened real world examples once they are available to make testing new plugins easier.

Data Types
----------

relation/list
^^^^^^^^^^^^^

The data contains a list of generic relations.


relation/distance
^^^^^^^^^^^^^^^^^

The relations encode distance between two entities.
Distance relations have a single numeric ``distance`` attribute.
Distances are unitless and must be positive or 0.

Additional attributes should be ignored.

Example:

.. code-block:: text

    source,target,distance
    entA,entB,0.7
    entB,entC,2
    entB,entB,0


relation/unit-distance
^^^^^^^^^^^^^^^^^^^^^^

Same as ``relation/distance``, but with an additional ``unit`` attribute.
The ``unit`` attribute gives the distance a specific unit, e.g., ``m`` for meters, ``km`` for kilometers, etc.
When converted to ``relation/distance``, all distances **must** first be converted to the same unit!

Additional attributes should be ignored.

.. code-block:: text

    source,target,distance,unit
    entA,entB,0.7,m
    entB,entC,2,m
    entB,entB,0,m


relation/similarity
^^^^^^^^^^^^^^^^^^^

The relations encode similarity between two entities.
Similarity relations have a single numeric ``similarity`` attribute.
Similarities are unitless and must be between 1 (for maximum similarity) or 0 (for no similarity) including both ends.
A relation from an entity to itself must always have a similarity of 1.

Additional attributes should be ignored.

Example:

.. code-block:: text

    source,target,similarity
    entA,entB,0.8
    entB,entC,0.2
    entB,entB,1
