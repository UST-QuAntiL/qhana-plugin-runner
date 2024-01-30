<!-- help me write a readme for the optimizer plugin -->
# Optimizer Plugin

This optimizer plugin is the result of the master thesis `Implementing Variational Quantum Algorithms as Compositions of Reusable Microservice-based Plugins` by Matthias Weilinger.

It is a plugin that can be used to optimize given data with the help of a user selected objective function and minimizer.

## Folder Structure (OUTDATED)

* `coordinator/` - Contains the coordinator plugin that is used to run the optimizer plugin.
* `objective_functions/` - Contains the objective function plugins.
* `minimizer/` - Contains the minimization plugins.

## How to create a new plugin for the optimizer

Detailed instructions on how to create new plugins can be found in the corresponding README files in the folders mentioned above.

* [Objective Function](objective_functions/README.md)
* [Minimizer](minimizer/README.md)

## How to create a generic plugin interaction

This section describes how to create a generic plugin interaction not just for the optimizer framework.

### Synchronous Interaction

For a synchronous interaction, just use the `requests` library to send a POST request to the plugin you want to interact with.

```python
from requests import post
response = post(url, json=data)
```

### Asynchronous Interaction

There are two ways for asynchronous interaction between plugins.
In any case a callback URL on the caller plugins side is needed to send the result of the interaction back to the caller plugin.

#### Plugin calls another plugin's microfrontend

The caller plugin should use the `invoke_task` function from the `interaction_utils.tasks` module.

```python
from interaction_utils.tasks import invoke_task

task = invoke_task.s(
    db_id=called_plugin_db_id,
    step_id="step_id",
    href=called_plugin_ui_href,
    ui_href=called_plugin_ui_href,
    callback_url=caller_plugin_callback_url,
    prog_value=0,
    task_log="log",
)
```

The called plugin's `ui_href` and `href` have to accept the `CallbackUrlSchema` as a query parameter.
The `CallbackUrlSchema` is defined in the `interaction_utils.schemas` module.

The called plugin's processing endpoint has to make callback to the `callback_url` with the result of the interaction.
It can use the `make_callback` function from the `interaction_utils.tasks` module.

```python
from interaction_utils.tasks import make_callback
make_callback(callback_url, result)
```

#### Plugin calls another plugin's long-running interaction endpoint

This is the case when the called plugin's processing endpoint is a long-running task other than a microfrontend.
The caller plugin makes a POST request to the called plugin's processing endpoint with the `callback_url` part of the request body.

```python
from requests import post
response = post(url, json=data)
```

The called endpoint schedules a celery task.
It places the `callback_url` and the `task_view` in the tasks database for the later callback mechanism.
The callback mechanism is triggered when the celery task is finished.

```python
db_task.data["status_changed_callback_urls"] = [callback_url]
db_task.data["task_view"] = url_for("tasks-api.TaskView", task_id=db_task.id, _external=True)
```

### Interaction Endpoints

This section describes how a plugin offers interaction endpoints to other plugins and how another plugin gets the url of the interaction endpoint.

#### Plugin offers an endpoint for other plugins to interact with

A plugin should offer an interaction endpoint for other plugins in the metadata endpoints.
The interaction endpoint has to have a type and a URL.

```python
from qhana_plugin_runner.api.plugin_schemas import EntryPoint, InteractionEndpoint, PluginMetadata

return PluginMetadata(
    ...
    entry_point=EntryPoint(
        interaction_endpoints=[
            InteractionEndpoint(
                type=type1,
                href=url2
            ),
            InteractionEndpoint(
                type=type2,
                href=url2
            ),
        ],
        ...
    ),
)
```

In case the interaction endpoint has a `task_id` as part of the URL, the `task_id` has to be represented as a string placeholder in the URL.
To do so use the `url_for_ie` function from the `interaction_utils.ie_utils` module.

```python
return PluginMetadata(
    ...
    entry_point=EntryPoint(
        interaction_endpoints=[
            InteractionEndpoint(
                type=type1,
                href=url_for_ie("endpoint")
            ),
        ],
        ...
    ),
)
```

#### Plugin gets the URL of another plugin's interaction endpoint

A plugin can get the URL of another plugin's interaction endpoint from the metadata endpoints.

```python
plugin_metadata: PluginMetadata = get_plugin_metadata(plugin_url)
interaction_endpoint_url = [
    element
    for element in plugin_metadata.entry_pointinteraction_endpoints
    if element.type == type
]
```

In case the interaction endpoint has a `task_id` as part of the URL, the `task_id` has to be replaced with the actual `task_id`.
To do so use the `ie_replace_task_id` function from the `interaction_utils.ie_utils` module.

```python
interaction_endpoint_url = ie_replace_task_id(interaction_endpoint_url, task_id)
```
