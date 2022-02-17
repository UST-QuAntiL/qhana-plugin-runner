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



Entities ``text/csv``
---------------------

Download: :download:`entities.csv <example_files/entities.csv>` 

.. code-block:: text

    ID,href,color
    paintA,example.com/paints/paintA,#8a2be2
    paintB,example.com/paints/paintA,#e9322d



Entities ``application/json``
-----------------------------

Download: :download:`entities.json <example_files/entities.json>` 

.. code-block:: json

    [
        {"ID": "paintA","href": "example.com/paints/paintA","color": "#8a2be2"},
        {"ID": "paintB","href": "example.com/paints/paintB","color": "#e9322d"}
    ]


Entities ``application/X-lines+json``
-------------------------------------

Download: :download:`entities-lines.json <example_files/entities-lines.json>` 

.. code-block:: json

    {"ID": "paintA","href": "example.com/paints/paintA","color": "#8a2be2"}
    {"ID": "paintB","href": "example.com/paints/paintB","color": "#e9322d"}



