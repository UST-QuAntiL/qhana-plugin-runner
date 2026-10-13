# Feature Engineering Pipeline Developer Documentation

## Plugin Description

The Feature Engineering Pipeline takes an entity list as input and provides dynamic processing options for each attribute of the entities, such as routing a tree taxonomy attribute through the Wu-Palmer pipeline. It aggregates the resulting feature vectors and optionally concatenates and reduces them using PCA.


### Inputs

| Input | Description | Required | Data Type | Content Type |
| ----- | ----------- | -------- | --------- | ------------ |
| `entitiesUrl` | URL to the entity list (e.g., subparts.csv). | Yes | `entity/list` | `application/json`, `application/X-lines+json`, `text/csv`  |
| `entitiesMetadataUrl` | URL to a file with the attribute metadata for the entities. | Yes | `entity/attribute-metadata` | `application/json`, `application/X-lines+json`, `text/csv` |
| `taxonomiesZipUrl` | URL to a zip file containing the taxonomies referenced by the metadata. | Yes | `graph/taxonomy` | `application/zip` |

### Outputs

| Output | Description | Optional | Data Type |
| ----- | ----------- | -------- | --------- |
| MDS Vectors | Final feature vectors produced by MDS. | No | `entity/vector` |
| Concatenated Vector | Combined vector of all successfully completed pipelines. | Yes | `entity/vector` |
| Dimension Mapping | Records which input file every dimension of the concatenated vector came from. | Yes | `entity/dimension-mapping` |
| Intermediate Results | Extracted element similarities, element distances, and attribute distances. | Yes | Various (e.g., `relation/element-similarities`) |
| PCA Outputs | Plot and metadata of the PCA reduction applied to the concatenated vector. | Yes | `custom/plot`, `custom/pca-metadata` |

### Plugins Used
**Wu-Palmer Pipeline:**
1. [Wu-Palmer](../all-plugins.md#wu-palmer-similarities-v025) 
2. [Similarities to Distances Transformer](../all-plugins.md#attribute-similarities-to-attribute-distances-transformers-v023) 
3. [Aggregator](../all-plugins.md#aggregators-v022) 
4. [MDS (Multidimensional Scaling)](../all-plugins.md#multidimensional-scaling-mds-v023)

**Mapping Pipeline:**
1. [Mapping to Distance](../all-plugins.md#taxanomy-mapping-to-distances-v030)
2. [Aggregator](../all-plugins.md#aggregators-v022) 
3. [MDS (Multidimensional Scaling)](../all-plugins.md#multidimensional-scaling-mds-v023)

**One-Hot Encoding Pipeline**
1. [One-Hot Encoding](../all-plugins.md#one-hot-encoding-v024)

**Finalization**
1. [Vector Concatination](../all-plugins.md#vector-concationation-plugin-v014)
2. [PCA (Principle Component Analysis)](../all-plugins.md#principle-component-analysis-pca-v023)


## Developing


### Adding new pipelines

To introduce a new pipeline to the routing architecture, you must register it in the schemas and update the orchestration logic.

1.  **Register in Schemas (`schemas.py`)**
    *   Define a constant for your pipeline (e.g., `NEW_PIPELINE_PLUGIN = "new_pipeline"`).
    *   Add the pipeline to the `PIPELINE_OPTIONS` dictionary.
    *   Map the pipeline's underlying microservice names to `PIPELINE_PLUGINS`.
2.  **Update the Routing UI**
    *   Ensure your pipeline is parsed correctly in `preprocessing_task` within `tasks.py` if it relies on specific taxonomy characteristics.
    *   The `RoutingStepFrontend` in `routes.py` will automatically pick up options added to `PIPELINE_OPTIONS` for the UI dropdowns.
3.  **Manage the Execution Queue (`tasks.py`)**
    *   Add your pipeline to the execution queue in `start_routing_task`.
    *   Update the `PIPELINE_STEP_COUNTS` and `PIPELINE_LOG_NAMES` dictionaries so the progress bar target tracks the correct number of sub-plugin steps.
    *   Ensure the finalization pipeline (`FINALIZE_PIPELINE`) always runs last to concatenate the outputs.
4.  **Route the Starting Step (`tasks_pipeline_steps.py`)**
    *   Adjust `launch_next_pipeline` to trigger the first Celery task of your new pipeline:
    ```python
    # Route to the correct starting step
    if next_pipeline == WU_PALMER_PLUGIN:
        task_data.add_task_log_entry("Starting Wu-Palmer Pipeline...")
        task_data.save(commit=True)
        start_wu_palmer.apply_async(args=[task_data.id])
    elif next_pipeline == NEW_PIPELINE_PLUGIN:
        task_data.add_task_log_entry("Starting New Pipeline...")
        task_data.save(commit=True)
        start_new_pipeline.apply_async(args=[task_data.id])
    ```
5.  **Handle Progression (`tasks.py`)**
    *   Update `handle_webhook_task` to route incoming webhooks for your pipeline to a custom progression handler.
    ```python
    elif current_pipeline == NEW_PIPELINE_PLUGIN:
        handle_new_pipeline_progression(task_data, db_id, source_url)
    ```
    *   Implement the handler to evaluate the `source_url` and trigger the subsequent steps sequentially:
    ```python
    def handle_new_pipeline_progression(task_data: ProcessingTask, db_id: int, source_url: str):
        if source_url == task_data.data.get(f"{NEW_PIPELINE_START}_url"):
            start_next_step.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
        elif source_url == task_data.data.get(f"{NEW_PIPELINE_END}_url"):
            finalize_pipeline.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    ```

### Adding or Adjusting Pipeline Plugins
When inserting a new sub-plugin into an existing or new pipeline, configure its schema inputs and define its Celery task.

1.  **Define Configuration Schemas (`schemas.py`)**
    *   Add plugin-specific settings to `PipelineSettingsSchema` (e.g., Enums, Integer fields).
    *   Group the settings in the `PIPELINE_SETTINGS_GROUPS` tuple to ensure they render in the UI.
    *   Note: Input parameters for the sub-plugin payloads must use camelCase (e.g., `entitiesMetadataUrl`) even if defined in snake_case (e.g., `entities_metadata_url`) in the runner schemas.
2.  **Implement the Celery Task (`tasks_pipeline_steps.py`)**
    *   Create a new function decorated with `@CELERY.task(bind=True, base=PipelineTask)`. The `PipelineTask` base handles transient network retries and fatal error logging.
    *   Load parameters using `load_params(task_data)` and execute the plugin using `run_pipeline_step`.
    ```python
    @CELERY.task(
        name=f"{FeatureEngineeringPipeline.instance.identifier}.start_new_plugin",
        bind=True,
        base=PipelineTask,
    )
    def start_new_plugin(self, db_id: int, source_url: str):
        task_data = ProcessingTask.get_by_id(db_id)
        params: InputParameters = load_params(task_data)
        
        # Extract output from previous step
        outputs = requests.get(source_url, timeout=REQUEST_TIMEOUT).json().get("outputs", [])
        previous_output_url = extract_output_url(outputs, "expected/data-type")
        
        payload = {
            "entitiesUrl": params.entities_url,
            "previousOutputUrl": previous_output_url,
            "customSetting": params.custom_setting,
        }

        run_pipeline_step(
            db_id=db_id,
            task_data=task_data,
            plugin_name=NEW_PLUGIN_NAME,
            logging_name="New Plugin",
            payload=payload,
        )
    ```
3.  **Persist Intermediate Results**
    *   If intermediate results should be saved, utilize `save_intermediate_results`:
    ```python
    if params.include_intermediate_results_in_output:
        save_intermediate_results(
            task_data=task_data,
            retries=self.request.retries,
            db_id=db_id,
            file=open_url(previous_output_url).content,
            file_name=f"{pipeline_label(task_data)}_output.zip",
            file_type="expected/data-type",
        )
    ```

---

### Handling Subscriptions and Responses

Plugin orchestration relies on the `run_pipeline_step` helper and robust webhook processing to prevent race conditions. (Also see [Advanced Plugin Techniques](../plugins-advanced.rst) for more detailed information)

*   **Subscription Initialization:** The `run_pipeline_step` function in `tasks_helpers.py` posts the payload to the sub-plugin, extracts the `Location` header to save the `active_subtask_url`, and sets up a webhook subscription via the `subscribe` utility. It also configures a polling watchdog (`monitor_result`) as a fallback in case the webhook fails.
*   **Webhook Processing (`tasks.py`):** The `handle_webhook_task` processes incoming webhooks (from `routes.py:WebhookView`) and watchdog events.
*   **Synchronization Guards:** Because network retries and the watchdog can cause duplicate completion events, `handle_webhook_task` uses `PluginState` database locks to ensure only one process progresses the pipeline for a given `source_url`. If a lock is already claimed, duplicate events are safely ignored.

## Tests

The testing suite validates schemas, individual pipeline steps, queue orchestration, and webhook synchronization. The tests rely heavily on Celery worker fixtures and an in-memory HTTP API stand-in called `PluginServer`.

### Test Data and Mocking
*   **PluginServer Stub:** Implemented in `data.py`, `PluginServer` intercepts HTTP requests using `monkeypatch`. It records payloads posted by `run_pipeline_step`, hands out fresh sub-task URLs, and serves mocked result statuses and outputs.
*   **Task Generation:** The `make_router_task` helper generates `ProcessingTask` instances in the exact database state the routing step would leave them, allowing tests to run isolated pipeline segments.

### Schema Validation (`test_schemas.py`)
*   Tests verify that valid payloads load successfully, optional fields default correctly, and missing or invalid fields (e.g., negative `mdsDimensions` or `pcaType` errors) are rejected.
*   **Enum Parity:** Tests dynamically import remote schemas (e.g., `mapping_distances.schemas`) to assert that the locally copied Enums match the source plugins perfectly, preventing unforeseen runtime breaks.

### Pipeline Steps (`test_pipeline_steps.py`)
*   **Payload Verification:** Step tests verify that payloads generated by tasks perfectly match the expected schema of the receiving sub-plugin.
*   **Result Forwarding:** Tests ensure that the `source_url` from a completed step is parsed to extract the necessary `href` to pass into the next plugin.
*   **Intermediate Results:** Tests confirm that intermediate results are correctly stored and zipped when the user requests them.

### Orchestration and Webhooks (`test_tasks.py` & `test_routes.py`)
*   **Full State Machine Runs:** Uses an `inline` fixture to execute Celery tasks synchronously, driving complete multi-pipeline runs from preprocessing to vector concatenation and PCA finalization.
*   **Synchronization Guards:** A dedicated test simulates a race condition by firing 10 concurrent webhook handlers (using `ThreadPoolExecutor`) for the same URL. It verifies that exactly one worker progresses the pipeline and 9 are rejected via the atomic database lock.
*   **Watchdog Recovery:** Tests confirm that if a webhook event is lost, a fallback `watchdog` poll successfully recovers the event and advances the pipeline.

### Task Helpers (`test_tasks_helpers.py`)
*   **Taxonomy Recommendations:** Verifies that `calculate_recommendations` correctly suggests mapping plugins for mapped taxonomies, Wu-Palmer for deep trees, and One-Hot encoding for flat trees.
*   **Error Handling:** Tests `PipelineTask` retry logic to ensure transient errors like `ConnectionError` and `Timeout` are automatically retried.