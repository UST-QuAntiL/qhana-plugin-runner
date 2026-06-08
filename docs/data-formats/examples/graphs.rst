Graphs (``graph/*``)
====================

+-----------------+--------------------------------------------------------------+
| data type       | graph/undirected, graph/directed, graph/dac, graph/tree,     |
|                 | graph/taxonomy                                               |
+-----------------+--------------------------------------------------------------+
| content types   | application/json                                             |
+-----------------+--------------------------------------------------------------+

Custom types can be used for data that has very few specific uses or that is only used by a small number of plugins.


Data Types
----------


graph/undirected
^^^^^^^^^^^^^^^^

The most generic graph type.
Edges must be interpreted without direction.
Edges may have a weight or any other attribute according to the definition of relations.


graph/directed
^^^^^^^^^^^^^^

A graph with directed edges.
Edges may have a weight or any other attribute according to the definition of relations.


graph/dac
^^^^^^^^^

Directed acyclic graph.
The graph is directed and must not contain any cycles.
Edges may have a weight or any other attribute according to the definition of relations.


graph/tree
^^^^^^^^^^

The graph is a tree structure.
Edges may have a weight or any other attribute according to the definition of relations.


graph/taxonomy
^^^^^^^^^^^^^^

Same as ``graph/tree``.

Taxonomy nodes carry an optional ``mapping`` field that assigns one or more
numeric values to a node. A mapping may hold a single value, several values, or
none at all. This lets downstream plugins attach numeric information to
categorical taxonomy values, for example a score, a weight, or a coordinate
vector.

Each node entity exposes two related fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - field
     - description
   * - ``mapping_raw``
     - The original mapping string as authored on the taxonomy item, preserved
       verbatim. Values are separated by spaces, for example ``"1"`` or
       ``"1 0 2.5"``. An empty string means no mapping was provided.
   * - ``mapping``
     - The parsed mapping as a list of floats. Empty tokens (from repeated or
       trailing spaces) are dropped. A missing or empty ``mapping_raw`` yields
       an empty list.

When at least one node in a taxonomy defines a mapping, all ``mapping`` lists
are padded to the same length so every node has the same number of values. The
length is that of the longest mapping in the taxonomy, and shorter lists are
right-padded with zeros. When no node defines a mapping, every ``mapping`` stays
an empty list and no padding is applied.

.. code-block:: json

   {
       "ID": "t_Gattung_3",
       "tax_item_name": "Sinfonie",
       "description": "",
       "mapping_raw": "1 0 2.5",
       "mapping": [1.0, 0.0, 2.5]
   }

