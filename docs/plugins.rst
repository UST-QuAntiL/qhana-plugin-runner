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

* ``./`` for plugin metadata and links to all the plugin endpoints

A plugin that does not follow that schema may not be usable from the QHAna UI later.


Plugin Metadata
---------------

Marshmallow schemas to render the plugin metadata can be found in the module :py:mod:`~qhana_plugin_runner.api.plugin_schemas`.

Example of plugin metadata:

.. code-block:: json
    :linenos:

    {
        "title": "Plugin Name (display title)",
        "description": "Human readable description",
        "name": "plugin-name",
        "version": "0.0.1",
        "type": "data-processor",
        "tags": [
            "example-tag",
            "preprocessing",
            "quantum-algorithm"
        ],
        "entryPoint": {
            "href": "./process/",
            "uiHref": "./ui/",
            "pluginDependencies": [
                {
                    "parameter": "helperPlugin",
                    "type": "processing",
                    "tags": ["my-helper", "!bad-tag"],
                    "required": true
                },
                {
                    "parameter": "extraHelperPlugin",
                    "name": "my-helper-plugin",
                    "version": ">=v0.1.0 <=v0.5.0"
                }
            ],
            "dataInput": [
                {
                    "parameter": "data",
                    "dataType": "entity/list",
                    "contentType": ["application/json", "text/csv"],
                    "required": true
                },
                {
                    "parameter": "extra",
                    "dataType": "some-other-type",
                    "contentType": ["*"]
                },
                {
                    "parameter": "text",
                    "dataType": "third-type",
                    "contentType": ["text/*"]
                }
            ],
            "dataOutput": [
                {"dataType": "output-type", "contentType": ["application/json"], "required": true}
            ]
        }
    }


.. list-table:: Plugin Metadata
    :header-rows: 1
    :widths: 25 30 45

    * - Name
      - Example
      - Description
    * - Title
      - My Awesome Plugin
      - Human readable title
    * - Description
      - Does something great
      - Human readable description
    * - Name
      - my-awesome-plugin
      - Stable machine readable name of the plugin. Must be URL-safe!
    * - Version
      - 0.0.1
      - A version conforming to <https://www.python.org/dev/peps/pep-0440/#public-version-identifiers>
    * - Type
      - ``processing`` | ``visalization`` | ``conversion``
      - A plugin that consumes data and creates new data is a ``processing`` plugin. 
        Plugins that consume data to produce a microfrontend visualization are ``visualization`` plugins.
        Plugins that consume input data in one format and output the data converted into a different format are ``conversion`` plugins.
        Conversion plugins work the same as processing plugins but must follow additional constraints.
        If support for conversion plugins is not implemented they must be treated as processing plugins.
    * - Tags
      - ``["data-loader", "MUSE"]``
      - A list of tags describing the plugin. Unknown tags must be ignored while parsing this list. 
        Tags specific to a certain plugin(-family) should be prefixed consistently to avoid name collisions.
    * - Entry Point
      - ``{…}``
      - The entry point of the plugin. Contains a link to the REST entry point and to the corresponding micro frontend.
    * - href
      - ./process/
      - The URL of the REST entry point resource.
    * - UI href
      - ./ui/
      - The URL of the micro frontend that corresponds to the REST entry point resource.
    * - Plugin Dependecies
      - ``[…]``
      - A list of plugin dependencies. Plugin dependencies can be specified by type (matching the plugin type), 
        tags (matching the plugin tags; ``!`` matches only if the tag is not present), name (matching the plugin name)
        and by version (matching an exact plugin version or a version range). A plugin must match for all attributes.
        Plugin dependencies are passed by reference (e.g. the URL to the plugin api root).
    * - Data Input
      - ``[…]``
      - A list of possible data inputs. Required data inputs must be provided other inputs are optional.
        The plugin should be selectable once all required data inputs can be provided from the experiment data store.
    * - Data Output
      - ``[…]``
      - A list of possible data outputs. Required data outputs will always be produced by the plugin.
    * - parameter
      - data
      - The parameter name (or key) under which the input data or plugin reference should be available.
    * - Data Type
      - entity/list
      - The data type tag associated with the data. Like content-type but for the data semantic.
    * - Content Type
      - ``["application/json"]``
      - Content type (or mimetype) of the data. Describes the encoding of the data.
        Exactly one of the given content types must match the actual content type of the data.

When specifying the accepted content or data type of a file input (or output) the following rules should be applied to match the specified type with the actual type:

  * ``something``, ``something/``, ``something/*`` are equivalent and only match anything before the ``/``
  * ``*`` matches anything
  * ``application/json`` is an exact match


Visualization Plugin Micro Frontend
-----------------------------------

A visualization plugin defines both ``href`` and ``hrefUi`` to point to the micro frontend that provides the data visualization.
The endpoint **must** accept a single query parameter ``data-url`` in the URL.
The accepted data type can be indicated by specifying a required dataInput.
A visualization plugin must have exactly one required data input or exactly one data input (that is implicitly assumed as required).
A visualization plugin **must not** produce any new data and **must not** list any data outputs.

.. note:: The specification of visualization plugins is WIP and will be finished later.


Processing Plugin Micro Frontend
--------------------------------

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


Communication With the Pluin Host
"""""""""""""""""""""""""""""""""

The Micro Frontends are loaded inside iframes.
This means that they are sandboxed from the parent window and need to communicate via messaging.
For this purpose a generic `microfrontend.js` is included in the static folder that is also part of the simple template.
Plugins that want to use this script should use the attributes described in the next section. 

The messages that can be exchanged with the plugin host are documented in an `AsyncAPI <https://www.asyncapi.com>`_ document.
The document can be found here :download:`asyncapi.json <asyncapi.json>`.
To view the document use the `AsyncAPI studio <https://studio.asyncapi.com/>`_.


Custom Attributes used in Micro Frontends
'''''''''''''''''''''''''''''''''''''''''

.. TODO: remove / rewrite this section

The Micro Frontend can use a number of custom html attributes to mark some inputs for the QHAna frontend to be enhanced.
This can be used to mark data input fields for the QHAna frontend.

.. list-table:: Custom Attributes
    :header-rows: 1
    :widths: 25 30 45

    * - Attribute
      - Example
      - Description
    * - ``data-input``
      - entity
      - Mark an input field as data input. The QHAna UI can choose to instrument the input with a datalist of possible data entries or with a data selection dialog.
    * - ``data-content-type``
      - ``application/json text/csv``
      - A list of acceptable content types seperated by a space.
    * - ``data-submit``
      - ``validate`` | ``submit``
      - Mark a submit button (or input) as validating or submitting.
        A validating button must point to a resource returning a validated micro frontend (possibly with extra error messages).
        A submitting button must point to the REST resource corresponding to the micro frontend.
        If this attribute is missing or unspecified a heuristic should be used to determine the type of the submit button.
    * - ``data-token``
      - ``ibmq``
      - Mark a password input as an API token input. The value specifies for which API the token will be used.
    * - ``data-private``
      - 
      - Mark an input as private. Values of private inputs must never be stored in permanent storage by QHAna. Password inputs are considered private by default.


Processing Plugin Results
-------------------------

The REST entry point of a plugin must return (or forward to) a valid plugin result value.

Example of a plugin result:

.. code-block:: json
    :linenos:

    {
        "status": "PENDING",
        "log": "…",
        "progress": {
            "value": 100,
            "start": 0,
            "target": 100,
            "unit": "%"
        },
        "steps": [
            {
                "href": ".../<UUID>/step1-process",
                "uiHref": ".../<UUID>/step1-ui",
                "stepId": "step1",
                "cleared": true
            },
            {
                "href": ".../<UUID>/step2b-process",
                "uiHref": ".../<UUID>/step2b-ui",
                "stepId": "step1.step2b",
                "cleared": true
            }
        ],
        "data": [
            {
                "href": ".../<UUID>/data/1",
                "dataType": "entity/list",
                "contentType": "application/json",
                "name": "EntityList"
            }
        ]
    }


.. list-table:: Result Attributes
    :header-rows: 1
    :widths: 25 30 45

    * - Name
      - Example
      - Description
    * - Status
      - ``PENDING`` | ``SUCCESS`` | ``ERROR``
      - The current state of the result. ``PENDING`` is for unfinished results that can be finished in the future.
        ``SUCCESS`` and ``ERROR`` are for finsihed results that were calculated successfully or produced an error.
    * - Log
      - Step 1: Finished processing 125 entities in 1.2 seconds.
      - Some human readable log of the result calculation. Use this field to convey errors that happened during the result calculation.
    * - Progress (optional)
      - ``{…}``
      - An object describing the current progress of the result calculation.
    * - Steps (optional)
      - ``[…]``
      - A (growing) list of sub-steps that need new (user-) input before the final result can be computed.
        Only the last step in the list can be marked with ``clear: false`` to indicate that the step is awaiting some input.
    * - Data
      - ``[…]``
      - The list of data that was produced for this result. Must only be present on ``SUCCESS`` or ``ERROR`` results.

Result Progress
"""""""""""""""

The result progress object can be used to indicate the current progress of a pending result.
If no progress object is given the progress is assumed to be indeterminate (e.g. a progress spinner should be displayed).
If a progress object is given then the progress can be displayed to the user (e.g. in form of a progress bar or a ``x/100 %`` counter).

.. list-table:: Result Progress
    :header-rows: 1
    :widths: 25 20 55

    * - Name
      - Example
      - Description
    * - Value
      - 70
      - The current progress value. Must be a number between ``start`` and ``target``.
    * - Start
      - 0
      - The starting progress value. Defines the point of no progress. Must be a number.
        If ``start`` is greater than ``target`` then the progress should be treated as a countdown type progress.
        By default progress counts up. Defaults to ``0`` if omitted.
    * - Target
      - 100
      - The target progress value that defines all work beeing finished. Must be a number. Defaults to ``100``.
    * - Unit (optional)
      - %
      - The unit the progress is given in. Can be used to display the progress to the user. Defaults to ``""``.

Result Steps
""""""""""""

Result steps are intermediate steps where additional input is required to continue the result computation.
The list of result steps should only grow with new steps added on the end of the list.
Only the last step should be active (e.g. not marked as cleared). Plugins that use multiple steps should store form inputs as usual in :py:attr:`~qhana_plugin_runner.db.models.tasks.ProcessingTask.parameters`. Data that is used in subsequent steps should then be extracted in the respective celery task and stored in the key-value store :py:attr:`~qhana_plugin_runner.db.models.tasks.ProcessingTask.data` that has dict-like functionality. Furthermore, whenever valid input data for the current uncleared step is available, :py:attr:`~qhana_plugin_runner.db.models.tasks.ProcessingTask.clear_previous_step` must be called in the function that handles the input data (i.e., the processing endpoint for the corresponding microfrontend endpoint).

.. list-table:: Result Steps
    :header-rows: 1
    :widths: 25 30 45

    * - Name
      - Example
      - Description
    * - href
      - http(s)://.../<UUID>/step1
      - A link to the REST resource accepting the input data for the step.
        This URL must be an absolute URL containing schema and host!
    * - UI href
      - http(s)://.../<UUID>/ui-step1
      - A link to the micro frontend corresponding to the REST resource accepting the input data for the step.
        This URL must be an absolute URL containing schema and host!
    * - Step ID (optional)
      - step1.step2b
      - A stable id corresponding to the current branch of the result computation. 
        The same choices in previous steps with the same data should always produce the same step id.
        The step id may be completely independent from the input data.
        The step id may be used to reliably repeat a recorded plugin interaction (or detect when the recorded interaction deviates from the current one).
    * - Cleared
      - ``true``
      - A flag indicating that the step has already accepted input and can be considered as cleared. Defaults to ``false`` if not specified.

Result Data
"""""""""""

The final result data is represented by a list of links to the data element.
The list must not be present until the result is completed.

.. list-table:: Result Data
    :header-rows: 1
    :widths: 25 30 45

    * - Name
      - Example
      - Description
    * - href
      - .../<UUID>/data/1
      - The URL where the (raw) data can be accessed.
    * - Name
      - FilteredEntityList
      - A human readable name given to the output data by the plugin. Should fit the data content.
    * - Content Type
      - application/json
      - The content type (mimetype) of the data. Describes how the data is encoded.
    * - Data Type
      - entity/list
      - The data type tag associated with the data. Describes what kind of data is encoded. Must not contain wildcards (``*``).

Conversion Plugins
------------------

Conversion plugins are special processing plugins.
The intended purpose of conversion plugins is to allow automatic conversion between different serialization formats.

.. note:: The specification of conversion plugins is WIP and will be finished later.


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
        description = "A plugin description."
        tags = ["tag"]
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
