Input Data Model
================

QHAna is to be used to analyze the data from the MUSE project.
The initial data model specified here is derived from the MUSE data model and provides a more generic representation of that data.



Attributes
----------

An attributes has a name and a value.
Attribute names are always strings, values can be of any type.
Attribute with the same name **must** have the same type.


Reserved Attributes
"""""""""""""""""""

The data model defines some special attributes.
These attributes must never be used as additional attributes (their attribute names are reserved).
This allows to use these attribute names to correctly determine if a given set of attributes is an entity, a relation or a graph.

.. note:: A list of reserved attribute names:

    .. hlist::
        :columns: 5
    
        * ``ID``
        * ``href``
        * ``source``
        * ``target``
        * ```GRAPH_ID```
        * ```entities```
        * ```relations```




Entities
--------

An entity is a collection of attributes with a unique ID.
The ID of an entity **must** be unique for all entities of the same type in one dataset.
Ideally the ID is unique in the whole dataset.
Every operation that transforms the data of an entity should preserve the ID so that the new data can be traced back to the original entity.
The value of the attribute **must** be a string.

An entity **should** have an ``href`` atribute that contains an URL pointing to the original location of the entity.
The attribute may be omitted for further processing of the entity data.
The value of the attribute **must** be a string.

An entity **can** have additional attributes.
All additional attributes **must** be either scalar values or a list (or set) of scalar values.

The following value types are considered scalar:

  * missing value (``null``/``None``)
  * boolean values (``true``/``false``)
  * numbers (``1``, ``1.34``)
  * strings (``"green"``)
  * ordinal values/enums (weekdays, months)
  * dates, times, datetimes, durations
  * locations/coordinates
  * referenced entity IDs (e.g. the id string of the referenced entity)


Example serializations of entities:
"""""""""""""""""""""""""""""""""""

JSON:

.. code-block:: json

    {
        "ID": "paintA",
        "href": "example.com/paints/paintA",
        "color": "#8a2be2"
    }

CSV:

.. code-block:: text

    ID,href,color
    paintA,example.com/paints/paintA,#8a2be2



Relations
---------

Relations can be used to model relations between entities without using entity attributes.
A relation is always directed and has a ``source`` and a ``target`` attribute.
They contain the IDs of the source/target entities.

Relations *do not* have an ID or a ``href`` attribute.

Like entites relations can also have additional attributes.
The same restrictions as for entity attributes apply.


Example serializations of relations:
""""""""""""""""""""""""""""""""""""

JSON:

.. code-block:: json

    {
        "source": "paintA",
        "target": "paintB"
    }

CSV:

.. code-block:: text

    source,target
    paintA,paintB



Graphs
------

A bundle of entities connected with relations can form a graph.
The graph must contain all entities and relations that make up the graph (e.g. no relation links to an entity that is not in the graph).
A graph may only reference entitites by their IDs.

A graph can have an GRAPH_ID with the same semantic as an entity ID.
The same rules as for entity IDs apply, however the GRAPH_ID of a graph **should** be globally unique (and not overlap with entity IDs).

A graph can have an optional ``type`` attribute.
The allowed values are ``undirected``, ``directed`` (the default), ``acyclic`` (implies ``directed``), ``tree`` and ``list`` (no relations).
Other values for type have no defined meening and should be ignored.
This implies that user defined graph types are allowed, but to be future proof user defined types should contain a ``-`` character.

The entities of the graph are stored in an attribute ``entities`` that can contain entity IDs or inline entity definitions.
Relations are always stored inline in the ``relations`` attribute.
Additionally an attribute ``ref-target`` can be specified on the graph to point to a file that contains the referenced entities.
The ref target attribute should contain the file name of that file.

Like entities graphs can also contain additional attributes.
In fact, leaving out the special ``entities`` and ``relations`` attributes graphs have the same features.


Example serializations of a graph:
""""""""""""""""""""""""""""""""""

JSON:

.. code-block:: json

    {
        "GRAPH_ID": "graphA",
        "type": "tree",
        "entities": [
            "paintA",
            {"ID": "paintB", "href": "example.com/paints/paintA", "color": "#e9322d"}
        ],
        "relations": [
            {"source": "paintA", "target": "paintB"}
        ]
    }



Executables
-----------

Executables are executable artifacts, e.g., source code.



Provenance
----------

Provenance data follows the same rules as entities but allows nested datastructures.
The provenance data type is used to describe the (future or past) execution of an executable artifact.



Custom Data Formats
-------------------

Custom data formats are completely free from any restrictions described for other data formats.
However, they should be used sparingly as their reuseability is limited.


