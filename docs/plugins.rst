Writing Plugins
===============


Plugin Code
-----------

A QHAna plugin is a python module or package that contains a class inheriting from :py:class:`~qhana_plugin_runner.util.plugins.QHAnaPluginBase`.
The plugin must be placed in a folder specified in the ``PLUGIN_FOLDERS`` config variable.
The plugin runner will import all plugins placed in the specified folders.
Only the root module of the plugin will be imported by the plugin runner, the plugin is responsible for importing the plugin implementation class and all celery tasks.
The root modules of plugins must have unique names to avoid problems on import!

A plugin may implement :py:meth:`~qhana_plugin_runner.util.plugins.QHAnaPluginBase.get_api_blueprint` to provide a set of API endpoints.
The returned Bluprint must be compatible with the flask smorest library.
The :py:class:`qhana_plugin_runner.api.util.SecurityBlueprint` is recommended for this purpose.

Plugins may implement :py:meth:`~qhana_plugin_runner.util.plugins.QHAnaPluginBase.get_requirements` to specify their requirements to the plugin runner (see :ref:`plugins:plugin dependencies`).



Plugin API Layout
-----------------

A plugin should expose the following endpoints with a blueprint:

* ``./`` for plugin metadata and links to all the plugin endpoints (format to be specified)
* ``./metadata/`` <- see ``./`` (moved to plugin root as extra endpoint is not needed)
* ``./ui/`` for the microfrontend that exposes the plugin parameters
* ``./process/`` for a processing resource

A plugin that does not follow that schema may not be usable from the QHAna UI later.


Plugin Metadata
---------------

Marshmallow schemas to render the plugin metadata can be found in the module :py:mod:`~qhana_plugin_runner.api.plugin_schemas`.

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


.. note:: The metadata format is work in progress and may change any time.


Plugin Micro Frontend
---------------------

All QHAna plugins should expose the parameters of the algorithm in a micro frontend (see :doc:`adr/0008-use-micro-frontends-to-expose-algorithm-parameters` for reasoning). 
The micro frontends should only use html and css.
Javascript can be used but should be used sparingly to ease the integration of the micro frontend into the QHAna UI later.

The parameters must be defined inside a native html form.
Starting the algorithm with the parameters must be done through a form submit button.

The plugin runner contains template macros that can be imported and used to auto generate form elements from simple marshmallow schemas.

.. code-block:: html+jinja
    :linenos:

    {% import 'forms.html' as forms %}

    <!-- process is the url of the processing resource, values the current form data or query data and errors are validation errors from marshmallow -->
    {% call forms.render_form(method='post') %}
        <!-- schema is the marshmallow schema and values is a dict containing prefilled (and serialized) values -->
        {{ forms.render_fields(schema, values=values, errors=errors) }}
        <div class="qhana-form-buttons">
        {{ forms.submit("validate")}}  <!-- validate form by sending it to the ui endpoint (should keep form inputs intact!) -->
        {{ forms.submit("submit", action=process)}}  <!-- submit data to processing resource -->
        </div>
    {% endcall %}



Plugin Dependencies
-------------------

Plugins can declare their external python dependencies by implementing the :py:meth:`~qhana_plugin_runner.util.plugins.QHAnaPluginBase.get_requirements` method.
The method must return the requirements in the same format as ``requirements.txt`` used by :program:`pip`.

.. seealso:: Requirements.txt format: https://pip.pypa.io/en/stable/cli/pip_install/#requirements-file-format

The plugin requirements of the loaded plugins can be installed using the :any:`plugin cli <cli:install>`.

.. important:: The installation will fail if **any** requirement cannot be satisfied.
    This includes the pinned requirements of the plugin runner itself!

    Plugin resolution may also take an exceptionally long time if the requirements have conflicting versions.
    Make sure that the plugin requirements are actually compatible with the plugin runner requirements.


.. note:: The requirement install mechanism is currently experimental and relies on the :program:`pip` resolver.
    This means that resolving complex requirement sets can take a very long time.
    Plugins should therfore minimize their requirements and (whenever possible) only depend on requirements installed by the plugin runner already.
    Requirements of the plugin runner should not be part of the requirements the plugin specifies itself.

.. warning:: Plugins must fail gracefully if their dependencies are not yet installed.

    If the plugin does not fail gracefully the plugin runner cannot get the plugin requirements by calling :py:meth:`~qhana_plugin_runner.util.plugins.QHAnaPluginBase.get_requirements`.
    This also means that it cannot install the requirements for that plugin!


Strategies for Plugins With External Dependecies
""""""""""""""""""""""""""""""""""""""""""""""""

Plugins with external dependencies must fail gracefully if their dependencies are not installed.
Otherwise they cannot inform the plugin runner about their dependencies.

Late Imports of Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of importing dependencies at the top of the module import your dependency locally (i.e. in the celery task instead of in the module).
This allows the plugin to load while the failing import does not get executed until the task is called.

This method is useful for one-module plugins that rely on external dependencies for specific calculations/functionality.

Catch import Errors
^^^^^^^^^^^^^^^^^^^

Surround the failing import with ``try``-``except`` and handle cases where the import failed gracefully.
A failing import can produce ``NameErrors`` when code tries to use the imported names.

This method is useful for one-module plugins that rely on external dependencies for specific calculations/functionality.

Reorganize Code
^^^^^^^^^^^^^^^

If the external dependency is tightly integrated into your plugin (e.g. through type hints) then it is best to move all code depending on the external functions into its own module or package.
This means that your plugin should be a python package!
Then one of the above techniques can be used to import that package.

Import in :py:meth:`~qhana_plugin_runner.util.plugins.QHAnaPluginBase.get_api_blueprint` method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a combination of all the above strategies.
The import happens late in the :py:meth:`~qhana_plugin_runner.util.plugins.QHAnaPluginBase.get_api_blueprint` method of the plugin.
To fail gracefully the import is guarded with a ``try``-``except`` statement.
The method is allowed to throw a ``NotImplementedError`` when the plugin does not provide a blueprint.

.. code-block:: python
    :emphasize-lines: 14-20

    from qhana_plugin_runner.util.plugins import QHAnaPluginBase

    ...

    class MyPlugin(QHAnaPluginBase):

        name = "my-plugin"
        version = "1.0"

        def __init__(self, app: Optional[Flask]) -> None:
            super().__init__(app)

        def get_api_blueprint(self):
            try:
                # late import, code was reorganized into submodule
                from .code_with_dependencies import MY_PLUGIN_BLP
                return MY_PLUGIN_BLP
            except ImportError:
                # fail gracefully with try-except block
                raise NotImplementedError("Plugin dependencies not installed.")



Long Running Tasks
------------------

Long running tasks can be implemented using :any:`Celery tasks <guide-tasks>`.
Task names should be unique.
This can be achieved by using the plugin name as part of the task name.

If a background task is started from a processing resource it must be registered in the database as a processing task (see ``plugins/hello_world.py``).
There are some utility tasks that can be used in the :py:mod:`~qhana_plugin_runner.tasks` module.


File Inputs
-----------

Plugins should load files from URLs (see ADR :doc:`adr/0009-always-pass-files-as-urls`).
The plugin runner provides a utility method (:py:func:`~qhana_plugin_runner.requests.open_url`) for accessing ``http(s)://``, ``file://`` and ``data:`` URLs.
If the plugin accepts large files then the URL should be opened with ``stream=True`` and the data should be read incrementally if possible.
This can reduce the memory footprint of the plugin.

Data formats for input files (especially those used by multiple plugins) should be specified in :doc:`data-formats/index`.
The plugin runner has builtin support for some formats, e.g. the ones specified in :doc:`data-formats/data-loader-formats`.

.. seealso:: The plugin utils module for marshalling entity data: :py:mod:`qhana_plugin_runner.plugin_utils.entity_marshalling`


File Outputs
------------

Plugins can use the FileStore :py:data:`~qhana_plugin_runner.storage.STORE` to persist intermediate files and result files.
The storage registry will forward methods to the configured default :py:class:`~qhana_plugin_runner.storage.FileStore`.
The plugin runner come with a file store implementation that uses the local filesystem as backend.

The final results of a task should be stored in the file store using the :py:meth:`~qhana_plugin_runner.storage.FileStore.persist_task_result` method.
If a task produces large intermediate results that have to be shared to following tasks then these results should be stored as a file using the :py:meth:`~qhana_plugin_runner.storage.FileStore.persist_task_temp_file` method.
The :py:class:`~qhana_plugin_runner.db.models.tasks.TaskFile` instance returned by that method should not be shared directly between tasks.
Instead share the :py:attr:`~qhana_plugin_runner.db.models.tasks.TaskFile.id` attribute and retrieve the task file info with :py:meth:`~qhana_plugin_runner.db.models.tasks.TaskFile.get_by_id`.

The files can be retrieved from the file store by requesting an URL for the file information.
Use :py:meth:`~qhana_plugin_runner.storage.FileStoreRegistry.get_task_file_url` for task files and :py:meth:`~qhana_plugin_runner.storage.FileStoreRegistry.get_file_url` for other files.
Tasks can use the internal URLs provided by these methods (set ``external=False``) while file downloads from outside of the plugin runner must use the external URLs.

Data formats for output files should be specified in :doc:`data-formats/index`.
The plugin runner has builtin support for some formats, e.g. the ones specified in :doc:`data-formats/data-loader-formats`.
When writing a new plugin that outputs data first consider using an already specified output format before creating your own.
This will increase the chance that other plugins can work with that data seamlessly.

.. seealso:: The plugin utils module for marshalling entity data: :py:mod:`qhana_plugin_runner.plugin_utils.entity_marshalling`
