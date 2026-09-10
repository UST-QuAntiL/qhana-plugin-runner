# Copyright 2026 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sqlite3 import IntegrityError

import requests
from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZipFile
from pathlib import PurePath
from celery.utils.log import get_task_logger
from sqlalchemy import update, insert
from sqlalchemy.exc import SQLAlchemyError

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.db import DB
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.db.models.virtual_plugins import PluginState
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import load_entities
from qhana_plugin_runner.requests import get_mimetype, open_url

from . import Router, ROUTER_BLP
from .schemas import (
    NONE_PLUGIN,
    WU_PALMER_PLUGIN,
    MAPPING_PLUGIN,
    ONE_HOT_PLUGIN,
    TRANSFORMERS_PLUGIN,
    AGGREGATOR_PLUGIN,
    MDS_PLUGIN,
    VECTOR_CONCAT_PLUGIN,
    PCA_PLUGIN,
    FINALIZE_PIPELINE,
    InputParameters,
    InputParametersSchema,
)
from .tasks_helpers import (
    CELERY_COUNTDOWN,
    REQUEST_TIMEOUT,
    PipelineTask,
    load_task,
    load_entity_attributes,
    log_task_event,
    taxonomy_ref,
    calculate_recommendations,
)
from .tasks_pipeline_steps import (
    launch_next_pipeline,
    start_transformers,
    start_aggregator,
    start_mds,
    finalize_pipeline,
    finalize_vector_concat,
    finalize_pca,
)

TASK_LOGGER = get_task_logger(__name__)


# --- Step 1: discover taxonomy attributes for the routing step ---
@CELERY.task(name=f"{Router.instance.identifier}.preprocessing_task", bind=True)
def preprocessing_task(self, db_id: int) -> str:
    """
    Discovers taxonomy attributes to populate the dynamic routing step UI.

    Downloads and parses the provided entities and metadata files to identify
    attributes that have a matching taxonomy in the provided ZIP file. It also
    parses the taxonomies to generate default pipeline recommendations
    (e.g., Mapping vs. Wu-Palmer) for the frontend.
    """

    TASK_LOGGER.info(f"Starting router preprocessing with db id '{db_id}'")
    task_data = load_task(db_id)

    params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    # The attribute metadata describes the full schema for all entity types. The
    # uploaded entities only contain a subset of those attributes, so collect the
    # attribute names actually present in the entities file.
    entity_attributes = load_entity_attributes(params.entities_url)

    # Collect the taxonomy present in the uploaded zip
    zip_content = open_url(params.taxonomies_zip_url).content
    taxonomies_zip = ZipFile(BytesIO(zip_content))
    available_taxonomies = set(taxonomies_zip.namelist())

    taxonomy_attributes = []
    recommendations = {}

    with open_url(params.entities_metadata_url) as response:
        mimetype = get_mimetype(response)
        for element in load_entities(response, mimetype):
            metadata = AttributeMetadata.from_dict(element)
            if metadata.ID not in entity_attributes:
                continue
            ref = taxonomy_ref(metadata)
            if ref:
                tax_filename = PurePath(ref).name
                # Ensure the taxonomy file exists in the zip
                matched_zip_path = next(
                    (
                        name
                        for name in available_taxonomies
                        if name.endswith(tax_filename)
                    ),
                    None,
                )

                if matched_zip_path:
                    taxonomy_attributes.append(metadata.ID)

                    recommendations[metadata.ID] = calculate_recommendations(
                        taxonomies_zip, matched_zip_path
                    )

    TASK_LOGGER.info(
        f"DEBUGGING: Found {len(taxonomy_attributes)} taxonomy attribute(s) with a matching "
        f"taxonomy in the zip: {taxonomy_attributes}"
    )

    task_data.data["taxonomy_attributes"] = taxonomy_attributes
    task_data.data["recommendations"] = recommendations
    task_data.save(commit=True)

    return f"Found {len(taxonomy_attributes)} taxonomy attribute(s)."


# --- Initial Routing Task Launcher ---
@CELERY.task(name=f"{Router.instance.identifier}.start_routing_task", bind=True)
def start_routing_task(self, db_id: int) -> str:
    """
    Translates user attribute selections into an execution queue and starts the first pipeline.

    Groups the attributes based on the user's selected pipeline (Wu-Palmer,
    Mapping, One-Hot, or None). It saves these groupings into the task data,
    generates a sequential queue of pipelines to execute, and calls
    `launch_next_pipeline` to begin execution.
    """

    task_data = ProcessingTask.get_by_id(id_=db_id)
    params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    selections = task_data.data.get("routing_selections", {})

    wu_palmer_attributes = [
        attr for attr, option in selections.items() if option == WU_PALMER_PLUGIN
    ]
    mapping_attributes = [
        attr for attr, option in selections.items() if option == MAPPING_PLUGIN
    ]
    one_hot_attributes = [
        attr for attr, option in selections.items() if option == ONE_HOT_PLUGIN
    ]
    none_selected = [attr for attr, option in selections.items() if option == NONE_PLUGIN]

    pipeline_queue = []
    total_plugins = 1  # 1, because the inital start value of progress_value has to be 1

    if wu_palmer_attributes:
        task_data.add_task_log_entry(
            f"Queued Wu-Palmer pipeline for attributes: {wu_palmer_attributes}"
        )
        task_data.data[f"{WU_PALMER_PLUGIN}_attributes"] = "\n".join(wu_palmer_attributes)
        pipeline_queue.append(WU_PALMER_PLUGIN)
        total_plugins += 4  # (Wu-Palmer, Transformers, Aggregator, MDS)

    if mapping_attributes:
        task_data.add_task_log_entry(
            f"Queued distances mapping pipeline for attributes: {mapping_attributes}"
        )
        task_data.data[f"{MAPPING_PLUGIN}_attributes"] = "\n".join(mapping_attributes)
        pipeline_queue.append(MAPPING_PLUGIN)
        total_plugins += 3  # (Mapping, Aggregator, MDS)

    if one_hot_attributes:
        task_data.add_task_log_entry(
            f"One-Hot encoding not yet supported. Selected One-Hot for attributes: {one_hot_attributes}"
        )

    if none_selected:
        task_data.add_task_log_entry(f"None selected attributes skipped: {none_selected}")

    if params.concat_output:
        total_plugins += 1  # (Vector Concatenation)
        if params.reduce_dimensions:
            total_plugins += 1  # (PCA)

    task_data.data["pipeline_queue"] = pipeline_queue
    task_data.data["current_pipeline"] = None

    task_data.progress_target = total_plugins
    task_data.progress_value = 1
    # Set to 1, because 0 did not work. Reason is a bug in UI repo. If the bug is fixed, this can probaply be set to 0 again.
    # Bug Location: https://github.com/UST-QuAntiL/qhana-ui/blob/8b3a32627cc39f5cd7b061ee6e9ed5a21c5c00dd/src/app/components/timeline-step/timeline-step.component.html

    task_data.progress_unit = "Steps"

    task_data.save(commit=True)

    # Trigger the first pipeline
    launch_next_pipeline(task_data)
    return "Routing task started and pipeline queued."


@CELERY.task(
    name=f"{Router.instance.identifier}.handle_webhook_task",
    bind=True,
    base=PipelineTask,
)
def handle_webhook_task(self, db_id: int, source_url: str, via: str):
    """
    Evaluates webhook payloads and triggers the next step in the active pipeline.

    Acts as the state machine's transition engine. It verifies the source URL,
    checks for duplicate execution events, and delegates the progression logic
    to the specific handler for the currently active pipeline (e.g., Wu-Palmer
    or Mapping).

    Attributes:
        db_id (int): The database ID of the processing task.
        source_url (str): The URL from which the webhook event originated.
        via (str): The method of delivery, either "webhook" or "watchdog".
            Webhook indicates a direct event from the sub-plugin, while watchdog indicates a recovery through polling after a missed webhook.
            Webhook and watchdog can trigger at the same time
    """

    task_data = ProcessingTask.get_by_id(db_id)

    # The webhook endpoint is unauthenticated, so a bogus event must never fail the task.
    if task_data is None:
        TASK_LOGGER.warning(f"Ignored webhook for the unknown task {db_id}.")
        return "Unknown task"

    known_urls = [
        task_data.data.get(f"{WU_PALMER_PLUGIN}_url"),
        task_data.data.get(f"{MAPPING_PLUGIN}_url"),
        task_data.data.get(f"{TRANSFORMERS_PLUGIN}_url"),
        task_data.data.get(f"{AGGREGATOR_PLUGIN}_url"),
        task_data.data.get(f"{MDS_PLUGIN}_url"),
        task_data.data.get(f"{VECTOR_CONCAT_PLUGIN}_url"),
        task_data.data.get(f"{PCA_PLUGIN}_url"),
    ]

    if not source_url or source_url not in known_urls:
        # A late watchdog poll for an already finished pipeline can land here during
        # normal runs, because launch_next_pipeline drops the step urls.
        TASK_LOGGER.warning(
            f"Ignored webhook for task {db_id} from an unknown source: {source_url!r}."
        )
        return "Unrecognized webhook source"

    delivery = via or "webhook"
    TASK_LOGGER.info(
        f"DEBUGGING: Completion event for {source_url} received (via={delivery})."
    )

    # remember the arrival before the (retryable) status request, so a slow handler
    # can still be told apart from an event that never arrived
    if delivery != "watchdog":
        webhook_seen = task_data.data.get("webhook_seen", {})
        if source_url not in webhook_seen:
            webhook_seen[source_url] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            task_data.data["webhook_seen"] = webhook_seen
            task_data.save(commit=True)

    sub_task_result = requests.get(source_url, timeout=REQUEST_TIMEOUT).json()
    status = sub_task_result.get("status", "PENDING")

    if status == "PENDING":
        return "Sub-task still pending"

    if status == "FAILURE":
        raise RuntimeError(
            f"Sub-plugin step failed ({source_url}). See the failed plugin's log for details."
        )

    # SNYCHRONIZATION GUARD: ensure that only one process progresses the pipeline for this source URL
    lock_key = f"router_sync_lock_{source_url}"
    plugin_id = ROUTER_BLP.name
    my_celery_id = self.request.id

    try:
        DB.session.execute(
            insert(PluginState).values(plugin_id=plugin_id, key=lock_key, value=0)
        )
        DB.session.commit()
        TASK_LOGGER.info(f"DEBUGGING: Lock {lock_key} created.")
    except Exception as e:
        DB.session.rollback()
        TASK_LOGGER.warning(
            f"DEBUGGING WARNING: Exception occured during lock creation for {lock_key}: {e}."
        )

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
        TASK_LOGGER.info(
            f"DEBUGGING: Duplicate completion event for {source_url} safely ignored (via={delivery})."
        )
        return "Sub-task already progressed"

    TASK_LOGGER.info(
        f"DEBUGGING: Lock for {lock_key} claimed by this process. Proceeding with progression."
    )

    # UPDATE JSON
    progressed_via = task_data.data.get("progressed_via", {})
    progressed_via[source_url] = delivery
    task_data.data["progressed_via"] = progressed_via
    task_data.save(commit=True)

    # WATCHDOG LOGGING
    if delivery == "watchdog":
        seen_at = task_data.data.get("webhook_seen", {}).get(source_url)
        if seen_at:
            log_task_event(
                task_data,
                f"Polling watchdog progressed {source_url} first. The webhook event "
                f"arrived at {seen_at}, but its handler was still retrying.",
                level="warning",
            )
        else:
            log_task_event(
                task_data,
                f"No webhook event arrived for {source_url}; the polling watchdog "
                "recovered the lost completion event.",
                level="warning",
            )

    # PROGRESS PIPELINE
    current_pipeline = task_data.data.get("current_pipeline")

    task_data.progress_value += 1
    task_data.save(commit=True)

    if current_pipeline == WU_PALMER_PLUGIN:
        handle_wu_palmer_progression(task_data, db_id, source_url)

    elif current_pipeline == MAPPING_PLUGIN:
        handle_mapping_progression(task_data, db_id, source_url)

    elif current_pipeline == FINALIZE_PIPELINE:
        handle_finalize_progression(task_data, db_id, source_url)

    else:
        return "Unrecognized pipeline state"


def handle_wu_palmer_progression(task_data: ProcessingTask, db_id: int, source_url: str):
    """
    Handle progression of the wu-palmer pipeline,
    by checking the source URL and triggering the next step in the sequence.
    """

    if source_url == task_data.data.get(f"{WU_PALMER_PLUGIN}_url"):
        start_transformers.apply_async(
            args=[db_id, source_url], countdown=CELERY_COUNTDOWN
        )
    elif source_url == task_data.data.get(f"{TRANSFORMERS_PLUGIN}_url"):
        start_aggregator.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get(f"{AGGREGATOR_PLUGIN}_url"):
        start_mds.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get(f"{MDS_PLUGIN}_url"):
        finalize_pipeline.apply_async(
            args=[db_id, source_url], countdown=CELERY_COUNTDOWN
        )


def handle_mapping_progression(task_data: ProcessingTask, db_id: int, source_url: str):
    """
    Handle progression of the mapping pipeline,
    by checking the source URL and triggering the next step in the sequence.
    """

    if source_url == task_data.data.get(f"{MAPPING_PLUGIN}_url"):
        start_aggregator.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get(f"{AGGREGATOR_PLUGIN}_url"):
        start_mds.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get(f"{MDS_PLUGIN}_url"):
        finalize_pipeline.apply_async(
            args=[db_id, source_url], countdown=CELERY_COUNTDOWN
        )


def handle_finalize_progression(task_data: ProcessingTask, db_id: int, source_url: str):
    """
    Handle progression of the finalize pipeline,
    by checking the source URL and triggering the next step in the sequence.
    """

    if source_url == task_data.data.get(f"{VECTOR_CONCAT_PLUGIN}_url"):
        finalize_vector_concat.apply_async(
            args=[db_id, source_url], countdown=CELERY_COUNTDOWN
        )
    elif source_url == task_data.data.get(f"{PCA_PLUGIN}_url"):
        finalize_pca.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
