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

Same as ``graph/tree``

