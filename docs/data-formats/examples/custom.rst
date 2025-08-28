Custom types (``custom/*``)
===========================

+-----------------+--------------------------------------------------------------+
| data type       | custom/*                                                     |
+-----------------+--------------------------------------------------------------+
| content types   | */*                                                          |
+-----------------+--------------------------------------------------------------+

Custom types can be used for data that has **very few specific uses** or that is **only used by a small number of plugins**.

.. warning:: Before creating a new custom format, first consider any of the existing formats for :ref:`entities <data-formats/data-model:entities>`, :ref:`relations <data-formats/data-model:relations>`, :ref:`graphs <data-formats/data-model:graphs>`, or :ref:`executables <data-formats/data-model:executables>`.
    Creating a new custom data type should be considered last.



Data Types
----------


custom/kernel-matrix
^^^^^^^^^^^^^^^^^^^^

.. TODO:: fill in information about custom data types in use


custom/clusters
^^^^^^^^^^^^^^^

.. TODO:: fill in information about custom data types in use


custom/attribute-distances
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. TODO:: fill in information about custom data types in use

.. TODO:: check if data type can be replaced by relation/distance


custom/attribute-similarities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. TODO:: fill in information about custom data types in use

.. TODO:: check if data type can be replaced by relation/similarity


custom/element-similarities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. TODO:: fill in information about custom data types in use

.. TODO:: check if data type can be replaced by relation/similarity


custom/entity-distances
^^^^^^^^^^^^^^^^^^^^^^^

.. TODO:: fill in information about custom data types in use

.. TODO:: check if data type can be replaced by relation/distance


custom/nisq-analyzer-result
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. TODO:: fill in information about custom data types in use


custom/pca-metadata
^^^^^^^^^^^^^^^^^^^

.. TODO:: fill in information about custom data types in use


custom/plot
^^^^^^^^^^^

Custom data type for presentational plots (e.g., html output from pyplot).

.. TODO:: maybe define new base data type for purely presentational data like plots, text, etc.


custom/hello-world-output
^^^^^^^^^^^^^^^^^^^^^^^^^

Custom data type for demo text output of the hello world plugins.


Current Non-Standard Custom Types
---------------------------------

This is a list of custom types that are not using the ``custom/`` prefix.

.. danger:: Do not use these types in new plugins!

* ``representative-circuit/*`` should be replaced with ``executable/circuit`` and a data name starting with ``representative-circuit``.
* ``plot/*`` should be replaced with ``custom/plot``
* ``txt/*`` should be replaced with ``custom/text``
* ``qnn-weights/*``
* ``vqc-metadata/*``
