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


Using existing Plugins in a Plugin Execution
--------------------------------------------

Plugins can make use of other plugins during their computation.
For this to work reliably, the used plugin must conform to some definition of an interface.
That is to say that plugins that should be usable by other plugins must be designed in a way that they can be used by the caller plugin in the first place.

.. seealso:: :doc:`/plugin-types/index` lists all interfaces currently defined as part of this documentation.



Starting a Processing Plugin
""""""""""""""""""""""""""""

Starting a procesing plugin can require arbitrary user inputs.
Such inputs are hard to impossible to automate reliably.
To avoid this there are two strategies:

1. Specify the (required) inputs for the starting step to enable automation.
   
   This approach is, for example, used by the circuit executor interface (see :doc:`/plugin-types/circuit-executor`).
2. Specify a special input that takes a webhook URL that is automatically subscribed to receive the relevant updates.
   
   This approach is, for example, used by the objective function interface (see :doc:`/plugin-types/objective-function`).

The second approach allows the calling plugin to add a step with the details of the plugin to be called while still being notified once this step is completed.
While the first approach can work with polling to receive the current task status, using the webhook subscription mechanism is a more efficient use of resources.
Additionally, the subscription mechanism allows for near instant notifications on such updates.


Subscribing to Task Result Updates
""""""""""""""""""""""""""""""""""

To avoid polling the task result resource, plugins can implement a subscription mechanism.
All the plugin has to do is provide a link with the ``subscription`` type in the ``links`` attribute of the task result (see :ref:`plugins:processing plugin results`).

.. note:: The plugin runner automatically implements this subscription mechanism for all plugins.

A plugin can then subscribe with a webhook to receive update events by issuing a post request to that link with the following JSON payload:

.. code-block:: json

    {
        "command": "subscribe",
        "event": "status",
        "webhookHref": "http://plugin.example.com/webhook/1234"
    }

Currently the plugin runner implements the following event types:


.. list-table:: Event Types
    :header-rows: 1
    :widths: 25 75

    * - Event
      - Description
    * - ``status``
      - The task status has changed (i.e., from ``PENDING`` to ``SUCCESS`` or ``FAILURE``).
    * - ``steps``
      - The list of steps was updated. Either a step was cleared, or a new step was added.
    * - ``details``
      - The task log or the progress was updated.

In case of an event, the webhook will be called as a post request with the following query parameters:


.. list-table:: Webhook Parameters
    :header-rows: 1
    :widths: 25 75

    * - Parameter
      - Description
    * - ``source``
      - The url of the task result resource that is the source of this event
    * - ``event``
      - The type of the event.

For any additional information, the plugin receiving the webhook notification must fetch the current task result resource.

Once the subscription is established, the calling plugin can add all steps of the called plugin to its own steps list.
This makes sure that the user will get to complete any unforseen step in both plugins.

.. warning:: Plugins that manually set the task state or update steps must make sure to also send the correct signals.
    Otherwise, the plugin runner is not able to notify the subscribed webhooks of the event!

    .. code-block:: python

        from flask.globals import current_app
        from qhana_plugin_runner.tasks import TASK_STATUS_CHANGED

        task_data: ProcessingTask
        # update task status
        ...
        task_data.save(commit=True)  # commit update to DB

        # send signal
        app = current_app._get_current_object()
        TASK_STATUS_CHANGED.send(app, task_id=task_data.id)

.. note:: The plugin runner contains utility functions to subscribe to plugins in the ``qhana_plugin_runner.plugin_utils.interop`` package.


.. todo:: extra links used for additional plugin interactions


