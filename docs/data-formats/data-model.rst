Input Data Model
================

QHAna is to be used to analyze the data from the MUSE project.
The initial data model specified here is derived from the MUSE data model and provides a more generic representation of that data.

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
        "id": "paintA",
        "href": "example.com/paints/paintA",
        "color": "#8a2be2"
    }

CSV:

.. code-block:: csv

   id,href,color
   paintA,example.com/paints/paintA,#8a2be2



Relations
---------




