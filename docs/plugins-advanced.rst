Advanced Plugin Techniques
==========================

.. attention:: Read :doc:`/plugins` first before reading about advanced techniques.


Creating Plugins at Runtime
---------------------------

The PluginRunner allows for the creation of plugins at runtime.
To register a new plugin create a new :py:class:`~qhana_plugin_runner.db.models.virtual_plugins.VirtualPlugin` object and persist it in the database.
This object contains all the information present in the plugin list resource of the PluginRunner.
Additionally it contains a parent identifier, which is the plugin identifier of the plugin that is managing this particular virtual plugin instance.


Signaling the Plugin Registry
"""""""""""""""""""""""""""""

The plugin registry only looks for new plugins in a specified intervall (the default is 15 minutes).
Thus, for newly registered plugins to show up immediately the plugin registry needs to be notified of the new plugin.
This can be done by sending the correct signal upon plugin creation.

.. code-block:: python

    from flask.globals import current_app
    from qhana_plugin_runner.db.models.virtual_plugins import VIRTUAL_PLUGIN_CREATED, VIRTUAL_PLUGIN_REMOVED

    # send this signal after the VirtualPlugin was saved to the database
    VIRTUAL_PLUGIN_CREATED.send(
        current_app._get_current_object(), plugin_url=plugin_url
    )

    # send this signal after a Virtual plugin was removed
    VIRTUAL_PLUGIN_REMOVED.send(
        current_app._get_current_object(), plugin_url=plugin_url
    )

.. note:: The signals must be sent with ``current_app._get_current_object()`` (i.e., the current app object, not the ``current_app`` proxy!).

The signals have subscribers setup that automatically notify the plugin registry of the new plugin.
For this to work, the app configuration must contain the URL of the plugin registry API.



Storing global State
--------------------

.. hint:: 
    
    This section is about storing data not related to a processing task. 
    Use :py:attr:`~qhana_plugin_runner.db.models.tasks.ProcessingTask.data` to store small data for ongoing processing tasks.

To store global state for plugins use the table :py:class:`~qhana_plugin_runner.db.models.virtual_plugins.PluginState`.
This class is intended to store state information for virtual plugins but can also be used by other plugins to store global state.

If a plugin needs to store larger documents in global state, then use the table :py:class:`~qhana_plugin_runner.db.models.virtual_plugins.DataBlob`.
This table is intended to store large data blobs in the persistent database.

.. warning:: 
    
    Do not store task results as :py:class:`~qhana_plugin_runner.db.models.virtual_plugins.DataBlob`.
    Use :py:const:`~qhana_plugin_runner.storage.STORE` to store file results instead.
