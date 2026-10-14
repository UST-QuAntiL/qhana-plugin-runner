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

Starting a processing plugin can require arbitrary user inputs.
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

There are two ways a calling plugin can subscribe to these updates: by issuing a direct HTTP request or by using the built-in Python utility function.

Method 1: Manual HTTP Subscription
''''''''''''''''''''''''''''''''''

A calling plugin (or external service) can subscribe to receive update events by issuing a POST request to the ``subscription`` link with the following JSON payload:

.. code-block:: json

    {
        "command": "subscribe",
        "event": "status",
        "webhookHref": "http://plugin.example.com/webhook/1234"
    }

The plugin runner's task API processes this command and registers a ``TaskUpdateSubscription`` in the database. To prevent redundant network calls, the API automatically ignores duplicate subscription requests for the same event type and webhook URL. 

To stop receiving notifications, the caller can send the same payload but with ``"command": "unsubscribe"``.

Method 2: Using the Interop Utility Function (Python)
'''''''''''''''''''''''''''''''''''''''''''''''''''''

For plugins built within the plugin runner ecosystem, you can abstract away the HTTP communication by using the ``subscribe`` function provided in the ``qhana_plugin_runner.plugin_utils.interop`` package. 

This utility automatically fetches the task result, extracts the subscription link, sends the HTTP POST request, and can optionally configure a Celery-based polling watchdog (``monitor_result``) to prevent lost events.
More information on the watchdog mechanism can be found in :ref:`watchdog-mechanism-ref`.

.. code-block:: python

    from qhana_plugin_runner.plugin_utils.interop import subscribe

    subscribed = subscribe(
        result_url=task_url,
        webhook_url="http://plugin.example.com/webhook/1234",
        events=["status"],
        check_for_updates=True,
        monitor_webhook_url="http://plugin.example.com/webhook/1234?via=watchdog"
    )

The ``subscribe`` function takes the following key arguments:

* ``result_url``: The URL of the sub-task's result resource.
* ``webhook_url``: The webhook URL in your calling plugin that will receive the event notifications.
* ``events``: A list of event types to subscribe to (or ``"all"``).
* ``check_for_updates``: Defaults to ``True``. If enabled, it spawns an asynchronous ``monitor_result`` task that polls the sub-plugin. If an update is detected, it triggers the webhook manually.
* ``monitor_webhook_url``: An alternative webhook URL used specifically by the watchdog. 
  This is useful for appending query parameters (e.g., ``?via=watchdog``) to track whether an event was delivered by the primary HTTP subscription or recovered by the watchdog.

.. note:: The :ref:`feature-engineering-pipeline` plugin demonstrates how to use this subscription mechanism for multiple plugins in practice.

Supported Event Types and Webhook Payload
'''''''''''''''''''''''''''''''''''''''''

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
This makes sure that the user will get to complete any unforeseen step in both plugins.

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


Robust Webhook Processing and Synchronization
"""""""""""""""""""""""""""""""""""""""""""""

When a calling plugin receives webhook events from multiple sub-plugins, it must be designed to handle duplicate or concurrent notifications gracefully to prevent race conditions.

* **Asynchronous Processing:** Webhook endpoints should acknowledge receipt immediately (e.g., HTTP 200) and offload the pipeline progression logic to an asynchronous background task.
* **Synchronization Guards:** Network retries or simultaneous polling fallbacks can cause the webhook handler to receive duplicate completion events for the exact same task. 
  Plugins should implement a database-level lock (such as a ``PluginState`` registry updated with the current worker's ID) tied to the ``source_url`` to ensure only a single process progresses the pipeline.
* **Source Verification:** The background webhook handler must verify the incoming ``source_url`` against the expected active sub-task URLs currently saved in the plugin's state. 
  If the URL is unrecognized or already cleared, the event should be safely ignored.

.. note:: The :ref:`feature-engineering-pipeline` plugin demonstrates how to handle multiple webhook calls.
    
    Approach: First update the database, then check if the current worker is the one, that did the update.

    .. code-block:: python

        # Last accessed: 2026.09.28
        def handle_webhook_task(self, db_id: int, source_url: str, via: str):
            # ...
            if not source_url or source_url not in known_urls:
                return "Unrecognized webhook source"
            # ...
            lock_key = f"router_sync_lock_{source_url}"
            plugin_id = FEATURE_ENGINEERING_PIPELINE_BLP.name
            my_celery_id = self.request.id

            try:
                DB.session.execute(insert(PluginState).values(plugin_id=plugin_id, key=lock_key, value=0))
                DB.session.commit()
            except Exception as e:
                DB.session.rollback()

            DB.session.execute(
                update(PluginState)
                .where(PluginState.plugin_id == plugin_id)
                .where(PluginState.key == lock_key)
                .where(PluginState.value == 0)
                .values(value=my_celery_id)
            )
            DB.session.commit()

            current_value = PluginState.get_value(plugin_id=plugin_id, key=lock_key)

            if current_value != my_celery_id:
                return "Sub-task already progressed"


.. _watchdog-mechanism-ref:
Implementing a Polling Watchdog (Fallback Mechanism)
""""""""""""""""""""""""""""""""""""""""""""""""""""

Relying exclusively on webhooks can lead to stalled pipelines if a network error occurs or an event is dropped.

* **Dual-Tracking Mechanism:** While subscribing to webhooks is the primary and most efficient notification method, caller plugins should simultaneously arm a polling watchdog as a safety net.
* **Event Provenance:** To differentiate between a standard webhook delivery and a watchdog recovery, append a query parameter like ``via=watchdog`` to the fallback monitor URL. 
  This allows the plugin to log when a primary webhook was missed and recover the lost event.

  .. code-block:: python

    # Example
    webhook_url = task_data.data["webhook_url"].replace("localhost", "127.0.0.1")
    monitor_url = webhook_url + ("&" if "?" in webhook_url else "?") + "via=watchdog"
* **Commit Before Subscribing:** When initiating a sub-plugin, the caller must extract the new task URL from the ``Location`` header and save it to the database before attempting to register the webhook subscription. 
  This prevents a race condition where a fast-completing sub-plugin fires a webhook before the caller plugin knows the expected URL.   
  
  .. code-block:: python

    # Example
    response = requests.post(plugin_url, data=payload, allow_redirects=False, timeout=REQUEST_TIMEOUT)
    task_url = urljoin(plugin_url, response.headers["Location"])
    task_data.data["active_subtask_url"] = task_url

.. note:: The :ref:`feature-engineering-pipeline` plugin demonstrates how to implement a polling watchdog.


State Machine and Pipeline Queues
"""""""""""""""""""""""""""""""""

Plugins that manage multi-step or dynamic processing pipelines must explicitly track their execution state to route webhook events to the correct next step.

* **Execution Queues:** Complex orchestration requires compiling a sequential execution queue and recording the currently active step in the task data before initiating any sub-plugins.
* **Idempotent Retries:** Network timeouts during sub-plugin initialization should trigger automatic retries. 
  To prevent spawning duplicate sub-tasks, the caller must check if a tracking URL for that specific step has already been stored before issuing a new POST request.
* **Targeted Progression:** Upon a successful webhook validation, the state machine should evaluate the ``current_pipeline`` state and the specific ``source_url`` to determine exactly which subsequent plugin step to trigger.
  
.. note:: The :ref:`feature-engineering-pipeline` plugin demonstrates how to handle a multi-step pipeline with a state machine and execution queue.


Using Additional Links
""""""""""""""""""""""

In some cases, only using the entry point of a plugin or the steps provided during execution is not sufficient to allow for certain kinds of interaction between plugins.
In those cases, plugins can provide additional links as part of their API surface exposed for such interactions.
The micro frontends of these plugins may also make use of these endpoints internally to expose this functionality to the user.

Plugins can expose two kinds of links:

1. Links that can be called **outside a task context**.
2. Links that require **task specific** information.

The first kind of links are links that can be useful to inquire more data before actually starting a plugin.
For example, a circuit executor may offer such a link to fetch the available quantum computers and their state prior to execution.
These links are specified in the plugin metadata (see :ref:`plugins:plugin metadata`).

The second kind of links can use task specific state for their computation.
For example, the :doc:`objective function plugins </plugin-types/objective-function>` expose such a link to allow calculating the loss value multiple times during the task execution.
These links should be specified in the ``links`` attribute of the task result resource (see :ref:`plugins:processing plugin results`).

