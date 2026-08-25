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

import json
import requests
from requests.exceptions import ConnectionError, Timeout
from celery.utils.log import get_task_logger
from flask.globals import current_app
from typing import Optional
from urllib.parse import urljoin
from zipfile import ZipFile

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask, TaskFile
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    EntityTupleMixin,
    load_entities,
)
from qhana_plugin_runner.plugin_utils.interop import (
    get_plugin_endpoint,
    monitor_result,
    subscribe,
)
from qhana_plugin_runner.requests import get_mimetype, open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import TASK_DETAILS_CHANGED, save_task_error

from .schemas import (
    WU_PALMER_PLUGIN,
    MAPPING_PLUGIN,
    InputParameters,
)

TASK_LOGGER = get_task_logger(__name__)
CELERY_COUNTDOWN = 3

REQUEST_TIMEOUT = 10


class PipelineTask(CELERY.Task):
    """Base task for router pipeline steps with centralized error handling.

    Transient network errors (dropped connections, timeouts) are retried
    automatically with exponential backoff. Any other exception fails the task
    loudly: the real traceback is logged to the console and the task log, and the
    processing task is marked ``FAILURE``.
    """

    autoretry_for = (ConnectionError, Timeout)
    retry_backoff = True
    retry_backoff_max = 60
    max_retries = 5

    @staticmethod
    def _get_db_id(args, kwargs) -> Optional[int]:
        if "db_id" in kwargs:
            return kwargs["db_id"]
        if args:
            return args[0]
        return None

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        db_id = self._get_db_id(args, kwargs)
        attempt = getattr(self.request, "retries", 0) + 1
        error_message = (
            f"Transient error in step '{self.name}' (db_id={db_id}); exception: {exc!r}\n"
            + f"Retry execution of '{self.name}' (attempt {attempt} of {self.max_retries})."
        )
        TASK_LOGGER.warning(error_message)
        if db_id is not None:
            try:
                # celery invokes on_retry outside of FlaskTask.__call__, so there is no app context
                with self.app.flask_app.app_context():
                    task_data = ProcessingTask.get_by_id(db_id)
                    if task_data is not None:
                        log_task_event(task_data, error_message, level="warning")
            except Exception as log_exc:  # Do not interrupt retry
                TASK_LOGGER.warning(
                    f"DEBUGGING: Could not record retry of step '{self.name}' (db_id={db_id}): {log_exc!r}"
                )
        super().on_retry(exc, task_id, args, kwargs, einfo)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        db_id = self._get_db_id(args, kwargs)
        if db_id is not None:
            save_task_error.delay(failing_task_id=task_id, db_id=db_id)
        else:
            TASK_LOGGER.error(
                f"DEBUGGING ERROR:Step '{self.name}' failed permanently: {exc!r}"
            )
        super().on_failure(exc, task_id, args, kwargs, einfo)


def log_task_event(task_data: ProcessingTask, message: str, level: str = "info"):
    """Log a message to the console and to the task log shown in the UI."""
    # TODO: Remove logging from ui and add it only to console.
    getattr(TASK_LOGGER, level)(message)
    task_data.add_task_log_entry(message)
    task_data.save(commit=True)
    app = current_app._get_current_object()
    TASK_DETAILS_CHANGED.send(app, task_id=task_data.id)


def extract_output_url(outputs: list, data_type: str) -> str:
    """Finds the URL of a specific output from a plugin result based on its dataType."""
    for output in outputs:
        if output.get("dataType") == data_type:
            return output["href"]
    raise ValueError(f"Could not find output with dataType {data_type}")


def plugin_process_url(task_data: ProcessingTask, plugin: str) -> str:
    """Resolve a pipeline plugin's process endpoint from its metadata url.

    The route handler stores the external metadata url per plugin in
    ``data["plugin_urls"]``. ``get_plugin_endpoint`` follows the metadata to
    the plugin's processing endpoint.
    """
    return get_plugin_endpoint(task_data.data["plugin_urls"][plugin])


def load_task(db_id: int) -> ProcessingTask:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)
    return task_data


def taxonomy_ref(metadata: AttributeMetadata) -> Optional[str]:
    """Return the referenced taxonomy file name, or ``None`` for non-taxonomy attributes.

    Taxonomy references point into the taxonomies zip, e.g.
    ``taxonomies.zip:t_Gattung.json`` (the file name is the part after the
    colon). Other references target plain files such as ``people.csv`` and are
    excluded. The ``refTarget`` field is the reliable signal here. The ``type``
    field carries the attribute id in the sample data, not the data type.
    """
    ref_target = metadata.ref_target or ""
    if ".zip:" not in ref_target:
        return None
    return ref_target.split(":", 1)[1]


def load_entity_attributes(entities_url: str) -> set:
    """Return the set of attribute names present in the entities file."""
    attributes: set = set()
    with open_url(entities_url) as response:
        mimetype = get_mimetype(response)
        for ent in load_entities(response, mimetype):
            if isinstance(ent, EntityTupleMixin):
                attributes.update(type(ent).entity_attributes)
                break
            attributes.update(ent.keys())
    return attributes


def calculate_recommendations(taxonomies_zip: ZipFile, zip_path: str) -> str:
    """
    Determines the optimal pipeline recommendation for a given taxonomy file.

    Reads the taxonomy JSON inside the provided ZIP file. If any entity contains
    non-empty 'mapping_raw' data, it recommends the Mapping plugin. Otherwise,
    it defaults to the Wu-Palmer plugin.
    """
    # TODO: Refine the recommendation detection (Template also allows for no recommendation)

    try:
        with taxonomies_zip.open(zip_path) as f:
            tax_data = json.load(f)
            # Check if any entity has a non-empty mapping_raw
            has_mapping = any(
                ent.get("mapping_raw", "") != "" for ent in tax_data.get("entities", [])
            )
            return MAPPING_PLUGIN if has_mapping else WU_PALMER_PLUGIN
    except Exception as e:
        TASK_LOGGER.warning(
            f"DEBUGGING WARNING: Could not read mapping for {zip_path}: {e}"
        )
        return WU_PALMER_PLUGIN


def run_pipeline_step(
    db_id: int,
    task_data: ProcessingTask,
    plugin_name: str,
    logging_name: str,
    payload: dict,
):
    """
    Start a pipeline sub-plugin and make sure its completion is observed.

    Error handling is delegated to :class:`PipelineTask`, which means that ransient
    errors are retried automatically, everything else fails the task loudly.

    Robustness:
    * The POST is skipped if the sub-task was already started for this step
      (``<plugin>_url`` already stored), so a retry never spawns a duplicate
      sub-task ("started twice").
    * Webhook subscription is the fast path, but a polling watchdog
      (``monitor_result``) is always armed as well, so a lost event cannot leave
      the pipeline stuck in ``PENDING``. Both deliver the same completion event;
      ``handle_webhook_task`` keeps the loser of that race a no-op.
    """
    TASK_LOGGER.info(f"DEBUGGING: Starting {logging_name} Step")

    existing_task_url = task_data.data.get(f"{plugin_name}_url")
    if existing_task_url:
        task_url = existing_task_url
        TASK_LOGGER.info(
            f"DEBUGGING: {logging_name} sub-task already started ({task_url}); skipping re-post."
        )
    else:
        plugin_url = plugin_process_url(task_data, plugin_name)
        response = requests.post(
            plugin_url, data=payload, allow_redirects=False, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()

        task_url = urljoin(plugin_url, response.headers["Location"])
        task_data.data[f"{plugin_name}_url"] = task_url

    webhook_url = task_data.data["webhook_url"].replace("localhost", "127.0.0.1")
    monitor_url = webhook_url + ("&" if "?" in webhook_url else "?") + "via=watchdog"

    try:
        subscribed = subscribe(
            result_url=task_url,
            webhook_url=webhook_url,
            events=["status"],
            monitor_countdown=CELERY_COUNTDOWN,
            monitor_webhook_url=monitor_url,
        )

        if subscribed:
            log_task_event(task_data, f"Subscribed to {logging_name} events.")
        else:
            log_task_event(
                task_data,
                f"Subscription for {logging_name} failed; relying on the polling watchdog.",
                level="warning",
            )
    except Exception as e:
        log_task_event(
            task_data,
            f"Subscription for {logging_name} failed ({e!r}); relying on the polling watchdog.",
            level="warning",
        )


def is_store_mds_output(params: InputParameters) -> bool:
    """
    Returns true if the MDS output should be stored or
    the output should be concatenated and the intermediate results shall be included.
    """
    if params.concat_output:
        return params.include_intermediate_results_in_output
    else:
        return True


def save_intermediate_results(
    task_data: ProcessingTask,
    retries: int,
    db_id: int,
    file: bytes,
    file_name: str,
    file_type: str,
    mimetype: str = "application/zip",
):
    existing_files = TaskFile.get_task_result_files(db_id)

    # Race condition for file_exists should be handled, by the synchronization check in handle_webhook_task.
    file_exists = any(f.file_name == file_name for f in existing_files)

    if not file_exists:
        # Normal Execution: File doesn't exist, proceed with saving
        STORE.persist_task_result(
            task_db_id=db_id,
            file_=file,
            file_name=file_name,
            file_type=file_type,
            mimetype=mimetype,
        )
        TASK_LOGGER.info(f"Successfully stored intermediate file: {file_name}")

    else:
        # Check the reason for duplicate. Either parallel execution or the task saved intermediate output, then fails and restarts and wants to save again.
        # In theory, parallel execution should no longer be possible.

        if retries > 0:
            msg = f"DEBUGGING: File {file_name} exists during retry {retries}. Skipping save."
            TASK_LOGGER.info(msg)
            task_data.add_task_log_entry(msg, commit=True)  # TODO: Remove this UI log
        else:
            error_msg = f"BUG/RACE CONDITION: Parallel execution detected! File {file_name} already exists on attempt 0."
            TASK_LOGGER.warning(error_msg)
            task_data.add_task_log_entry(error_msg, commit=True)


def has_enough_pca_dimensions(
    task_data: ProcessingTask, params: InputParameters, vector_response: requests.Response
) -> bool:
    mimetype = get_mimetype(vector_response)
    if mimetype is None:
        return False

    for ent in load_entities(vector_response, mimetype):
        if isinstance(ent, EntityTupleMixin):
            attributes = set(type(ent).entity_attributes)
        else:
            attributes = set(ent.keys())

        dimensions = len(attributes - {"ID", "href"})
        has_more_dimensions = dimensions > params.pca_dimensions

        if not has_more_dimensions:
            info_msg = f"PCA reduction step disabled and skipped, because vector has {dimensions} dimensions, which is not more than requested {params.pca_dimensions}."
            TASK_LOGGER.info(info_msg)
            task_data.add_task_log_entry(info_msg, commit=True)

        return has_more_dimensions

    return False
