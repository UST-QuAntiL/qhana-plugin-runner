Data Formats defined for QHAna Plugins
======================================


A collection of data (serialization) formats used by QHAna plugins to exchange data.

.. note:: Plugins consume and produce serialized data. 
    There are a few considerations when choosing a suitable serialization format:

      * Will the file contain many individual data instances (e.g. entities or relations)?
      * Can the target format support streaming reads/writes?
      * How can certain datatypes be serialized in the target format (e.g. json does not support datetime)?
      * Is the target format standardized?
      * What tools/libraries exist to work with the target format?

    Also make sure to check whether a similar format is already defined before defining a new serialization format.

.. toctree::
    :name: formats-toc

    data-model
    data-loader-formats
    recipes
    examples/index
