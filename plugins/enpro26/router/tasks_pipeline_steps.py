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

import traceback
import requests
from urllib.parse import urljoin
from celery.utils.log import get_task_logger
from marshmallow import EXCLUDE

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import Router
from .schemas import InputParameters, InputParametersSchema
from .tasks_helpers import subscribe_to_plugin, extract_output_url, plugin_process_url

TASK_LOGGER = get_task_logger(__name__)
CELERY_COUNTDOWN = 3

# Collection of celery tasks for starting plugins that are substeps of the various pipelines managed by the routing plugin.

# --- BOILERPLATE HELPERS ---


def run_pipeline_step(celery_task, db_id: int, step_log_name: str, step_action: callable):
    """Boilerplate wrapper to handle task fetching, execution, retries, and error logging."""
    TASK_LOGGER.info(f"Starting {step_log_name} Step")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        step_action(task_data)
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        task_data.add_task_log_entry(
            f"Error occurred during {step_log_name}. Attempting to retry.\nError Message: {e}"
        )
        task_data.save(commit=True)
        raise celery_task.retry(exc=e, countdown=CELERY_COUNTDOWN)
    except Exception as e:
        task_data.add_task_log_entry(
            f"CRASH in {step_log_name} step:\n{traceback.format_exc()}"
        )
        save_task_error.delay(failing_task_id=celery_task.request.id, db_id=db_id)


def invoke_plugin_and_subscribe(
    task_data: ProcessingTask,
    plugin_name: str,
    payload: dict,
    step_name: str,
    url_key: str,
):
    """Boilerplate wrapper to construct the request, launch the plugin, and subscribe to the webhook."""
    plugin_url = plugin_process_url(task_data, plugin_name)
    response = requests.post(plugin_url, data=payload, allow_redirects=False)
    response.raise_for_status()

    task_url = urljoin(plugin_url, response.headers["Location"])
    task_data.data[url_key] = task_url
    task_data.add_task_log_entry(f"Started {step_name}.")
    task_data.save(commit=True)

    subscribe_to_plugin(task_url, task_data.data["webhook_url"])


# --- PIPELINE ORCHESTRATION ---


def launch_next_pipeline(task_data: ProcessingTask):
    """Pops the next pipeline from the queue and triggers it."""
    queue = task_data.data.get("pipeline_queue", [])

    if not queue:
        # All pipelines have been finished.
        # TODO: Add Dimension reduction and vector merge here
        save_task_result.delay("All Pipelines Completed Successfully!", task_data.id)
        return

    # Pop the next pipeline and update state
    next_pipeline = queue.pop(0)
    task_data.data["pipeline_queue"] = queue
    task_data.data["current_pipeline"] = next_pipeline

    # Route to the correct starting step
    if next_pipeline == "wu_palmer":
        task_data.add_task_log_entry(
            "Starting Wu-Palmer Pipeline. Includes: Wu-Palmer, Transformer, Aggregator, MDS"
        )
        task_data.save(commit=True)
        start_wu_palmer.apply_async(args=[task_data.id])
    elif next_pipeline == "mapping":
        task_data.add_task_log_entry(
            "Starting Mapping Pipeline. Includes: Mapping-Distances, Aggregator, MDS"
        )
        task_data.save(commit=True)
        start_mapping.apply_async(args=[task_data.id])


# --- INDIVIDUAL TASK STEPS ---


# --- WU-PALMER TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.start_wu_palmer", bind=True, max_retries=5
)
def start_wu_palmer(self, db_id: int):
    """Starting the Wu-Palmer plugin"""

    def action(task_data):
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters
        )
        payload = {
            "entitiesUrl": params.entities_url,
            "entitiesMetadataUrl": params.entities_metadata_url,
            "taxonomiesZipUrl": params.taxonomies_zip_url,
            "attributes": task_data.data["wu_palmer_attributes"],
            "root_is_part_of_hierarchy": str(params.root_is_part_of_hierarchy).lower(),
        }
        invoke_plugin_and_subscribe(
            task_data, "wu_palmer", payload, "Wu-Palmer", "wu_palmer_url"
        )

    run_pipeline_step(self, db_id, "Wu-Palmer", action)


# --- MAPPING TASK ---
@CELERY.task(name=f"{Router.instance.identifier}.start_mapping", bind=True, max_retries=5)
def start_mapping(self, db_id: int):
    """Starting the mapping-distances plugin"""

    def action(task_data):
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters
        )
        payload = {
            "entitiesUrl": params.entities_url,
            "entitiesMetadataUrl": params.entities_metadata_url,
            "taxonomiesZipUrl": params.taxonomies_zip_url,
            "attributes": task_data.data["mapping_attributes"],
            "distanceMetric": params.distance_metric.name,
        }
        invoke_plugin_and_subscribe(
            task_data, "mapping", payload, "Mapping Distances", "mapping_url"
        )

    run_pipeline_step(self, db_id, "Mapping", action)


# --- TRANSFORMER TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.start_transformers",
    bind=True,
    max_retries=5,
)
def start_transformers(self, db_id: int, source_url: str):
    """Starting the element_sim-to-element_dist-transformers plugin"""

    def action(task_data):
        outputs = requests.get(source_url).json().get("outputs", [])
        element_sims_url = extract_output_url(outputs, "relation/element-similarities")
        params = InputParametersSchema(unknown=EXCLUDE).loads(task_data.parameters)

        if params.include_intermediate_results_in_output:
            STORE.persist_task_result(
                db_id,
                open_url(element_sims_url).content,
                "wu_palmer_similarities.zip",
                "relation/element-similarities",
                "application/zip",
            )

        payload = {
            "similaritiesUrl": element_sims_url,
            "attributes": task_data.data["wu_palmer_attributes"],
            "transformer": params.transformer.name,
        }
        invoke_plugin_and_subscribe(
            task_data, "transformers", payload, "Transformers", "transformers_url"
        )

    run_pipeline_step(self, db_id, "Transformers", action)


# --- AGGREGATOR TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.start_aggregator",
    bind=True,
    max_retries=5,
)
def start_aggregator(self, db_id: int, source_url: str):
    """Starting the attribute-distance-aggregator plugin"""

    def action(task_data):
        outputs = requests.get(source_url).json().get("outputs", [])
        element_dists_url = extract_output_url(outputs, "relation/element-distances")
        params = InputParametersSchema(unknown=EXCLUDE).loads(task_data.parameters)

        if params.include_intermediate_results_in_output:
            prefix = task_data.data.get("current_pipeline", "unknown")
            STORE.persist_task_result(
                db_id,
                open_url(element_dists_url).content,
                f"{prefix}_element_distances.zip",
                "relation/element-distances",
                "application/zip",
            )

        payload = {
            "entitiesUrl": params.entities_url,
            "elementDistancesUrl": element_dists_url,
        }
        invoke_plugin_and_subscribe(
            task_data, "aggregator", payload, "Aggregator", "aggregators_url"
        )

    run_pipeline_step(self, db_id, "Aggregator", action)


# --- MDS TASK ---
@CELERY.task(name=f"{Router.instance.identifier}.start_mds", bind=True, max_retries=5)
def start_mds(self, db_id: int, source_url: str):
    """Starting the attribute-distance-mds plugin"""

    def action(task_data):
        outputs = requests.get(source_url).json().get("outputs", [])
        attr_dists_url = extract_output_url(outputs, "relation/attribute-distances")
        params = InputParametersSchema(unknown=EXCLUDE).loads(task_data.parameters)

        if params.include_intermediate_results_in_output:
            prefix = task_data.data.get("current_pipeline", "unknown")
            STORE.persist_task_result(
                db_id,
                open_url(attr_dists_url).content,
                f"{prefix}_attribute_distances.zip",
                "relation/attribute-distances",
                "application/zip",
            )

        payload = {
            "attributeDistancesUrl": attr_dists_url,
            "dimensions": params.dimensions,
            "metric": params.metric.name,
            "nInit": params.n_init,
            "maxIter": params.max_iter,
            "missingDataHandling": params.missing_data_handling.name,
        }
        invoke_plugin_and_subscribe(task_data, "mds", payload, "MDS", "mds_url")

    run_pipeline_step(self, db_id, "MDS", action)


# --- END OF PIPELINE ---
@CELERY.task(
    name=f"{Router.instance.identifier}.finalize_pipeline", bind=True, max_retries=5
)
def finalize_pipeline(self, db_id: int, source_url: str):
    """Finalizing the pipeline. Writing the final vector file to output"""

    def action(task_data):
        outputs = requests.get(source_url).json().get("outputs", [])
        final_dists_url = extract_output_url(outputs, "entity/vector")

        prefix = task_data.data.get("current_pipeline", "unknown")
        STORE.persist_task_result(
            db_id,
            open_url(final_dists_url).content,
            f"{prefix}_mds_final_vectors.zip",
            "entity/vector",
            "application/zip",
        )
        launch_next_pipeline(task_data)

    run_pipeline_step(self, db_id, "Finalize Pipeline", action)
