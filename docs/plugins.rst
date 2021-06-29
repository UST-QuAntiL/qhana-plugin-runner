Writing Plugins
===============


Plugin API Layout
-----------------

A plugin should expose the following endpoints with a blueprint:

* ``./`` for links to all the plugin endpoints (format to be specified)
* ``./metadata/`` for plugin metadata (format to be specified)
* ``./ui/`` for the microfrontend that exposes the plugin parameters
* ``./process/`` for a processing resource



Plugin Metadata
---------------

First draft of example plugin metadata:

.. code-block:: json
    :linenos:

    {
        "title": "Plugin Name (display title)",
        "description": "Human readable description",
        "name": "plugin-name",
        "version": "0.0.1",
        "type": "data-processor",
        "tags": [
            "namespace:example-tag",
            "ml:preprocessing",
            "q:quantum-algorithm"
        ],
        "processingResourceMetadata": {
            "href": "./process/",
            "uiHref": "./ui/",
            "fileInputs": [
                [
                    {"type": "point-list", "content-type": "application/json", "schema": "..."}, 
                    {"type": "label-list", "content-type": "application/json", "schema": "..."}
                ],
                [{"type": "labeled-points", "content-type": "application/json", "schema": "..."}]
            ],
            "fileOutputs": [
                {"type": "trained-model-parameters", "content-type": "application/json", "schema": "..."}
            ]
        }
    }


