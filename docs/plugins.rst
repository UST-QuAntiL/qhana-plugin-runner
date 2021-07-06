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



Plugin Micro Frontend
---------------------

All QHAna plugins should expose the parameters of the algorithm in a micro frontend (see :doc:`adr/0008-use-micro-frontends-to-expose-algorithm-parameters` for reasoning). 
The micro frontends should only use html and css.
Javascript can be used but should be used sparingly to ease the integration of the micro frontend into the QHAna UI later.

The parameters must be defined inside a native html form.
Starting the algorithm with the parameters must be done through a form submit button.

The plugin runner contains template macros that can be imported and used to auto generate form elements from simple marshmallow schemas.

.. code-block:: jinja
    :linenos:

    {% import 'forms.html' as forms %}

    <!-- process is the url of the processing resource -->
    {% call forms.render_form(action=process, method='post') %}
        <!-- schema is the marshmallow schema and values is a dict containing prefilled (and serialized) values -->
        {{ forms.render_fields(schema, values=values) }}
        {{ forms.submit("submit")}}
    {% endcall %}



Plugin Dependencies
-------------------

Plugins can declare their external python dependencies by implementing the :py:meth:`~qhana_plugin_runner.util.plugins.QHAnaPluginBase.get_requirements` method.
The method must return the requirements in the same format as ``requirements.txt`` used by :program:`pip`.

.. seealso:: Requirements.txt format: https://pip.pypa.io/en/stable/cli/pip_install/#requirements-file-format

The plugin requirements of the loaded plugins can be installed using the :any:`plugin cli <cli:install>`.

.. important:: The installation will fail if **any** requirement cannot be satisfied.
    This includes the pinned requirements of the plugin runner itself!


.. note:: The requirement install mechanism is currently experimental and relies on the :program:`pip` resolver.
    This means that resolving complex requirement sets can take a very long time.
    Plugins should therfore minimize their requirements and (whenever possible) only depend on requirements installed by the plugin runner already.
    Requirements of the plugin runner should not be part of the requirements the plugin specifies itself.
