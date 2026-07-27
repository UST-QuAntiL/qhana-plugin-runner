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
from celery.utils.log import get_task_logger
from marshmallow import EXCLUDE

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import Router
from .schemas import (
    WU_PALMER_PLUGIN,
    MAPPING_PLUGIN,
    TRANSFORMERS_PLUGIN,
    AGGREGATOR_PLUGIN,
    MDS_PLUGIN,
    VECTOR_CONCAT_PLUGIN,
    InputParameters,
    InputParametersSchema,
)
from .tasks_helpers import (
    CELERY_COUNTDOWN,
    extract_output_url,
    is_store_mds_output,
    run_pipeline_step,
)

TASK_LOGGER = get_task_logger(__name__)


# --- PIPELINE ORCHESTRATION ---
def launch_next_pipeline(task_data: ProcessingTask):
    """Pops the next pipeline from the queue and triggers it."""
    queue = task_data.data.get("pipeline_queue", [])

    if not queue:
        # TODO: Add Dimension reduction here
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters
        )
        if params.concat_output:
            task_data.add_task_log_entry(
                "Starting Vector concat plugin after all pipelines completed successfully."
            )
            task_data.save(commit=True)
            start_vector_concat.apply_async(args=[task_data.id])

        save_task_result.delay("All Pipelines Completed Successfully!", task_data.id)
        return

    # Pop the next pipeline and update state
    next_pipeline = queue.pop(0)
    task_data.data["pipeline_queue"] = queue
    task_data.data["current_pipeline"] = next_pipeline

    # Route to the correct starting step
    if next_pipeline == WU_PALMER_PLUGIN:
        task_data.add_task_log_entry(
            "Starting Wu-Palmer Pipeline. Includes: Wu-Palmer, Transformer, Aggregator, MDS"
        )
        task_data.save(commit=True)
        start_wu_palmer.apply_async(args=[task_data.id])
    elif next_pipeline == MAPPING_PLUGIN:
        task_data.add_task_log_entry(
            "Starting Mapping Pipeline. Includes: Mapping-Distances, Aggregator, MDS"
        )
        task_data.save(commit=True)
        start_mapping.apply_async(args=[task_data.id])


"""Collection of celery tasks for starting plugins that are substeps of the various pipelines managed by the routing plugin."""


# --- WU-PALMER TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.start_wu_palmer", bind=True, max_retries=5
)
def start_wu_palmer(self, db_id: int):
    """Starting the Wu-Palmer plugin"""
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters
        )
        payload = {
            "entitiesUrl": params.entities_url,
            "entitiesMetadataUrl": params.entities_metadata_url,
            "taxonomiesZipUrl": params.taxonomies_zip_url,
            "attributes": task_data.data[f"{WU_PALMER_PLUGIN}_attributes"],
            "root_is_part_of_hierarchy": str(params.root_is_part_of_hierarchy).lower(),
        }
    except Exception as e:
        task_data.add_task_log_entry(
            f"CRASH in Wu-Palmer during input loading:\n{traceback.format_exc()}"
        )
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)
        return

    run_pipeline_step(
        celery_task=self,
        db_id=db_id,
        task_data=task_data,
        plugin_name=WU_PALMER_PLUGIN,
        logging_name="Wu-Palmer",
        payload=payload,
    )


# --- MAPPING TASK ---
@CELERY.task(name=f"{Router.instance.identifier}.start_mapping", bind=True, max_retries=5)
def start_mapping(self, db_id: int):
    """Starting the mapping-distances plugin"""
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters
        )
        payload = {
            "entitiesUrl": params.entities_url,
            "entitiesMetadataUrl": params.entities_metadata_url,
            "taxonomiesZipUrl": params.taxonomies_zip_url,
            "attributes": task_data.data[f"{MAPPING_PLUGIN}_attributes"],
            "distanceMetric": params.distance_metric.name,
        }
    except Exception as e:
        task_data.add_task_log_entry(
            f"CRASH in Mapping during input loading:\n{traceback.format_exc()}"
        )
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)
        return

    run_pipeline_step(
        celery_task=self,
        db_id=db_id,
        task_data=task_data,
        plugin_name=MAPPING_PLUGIN,
        logging_name="Mapping Distances",
        payload=payload,
    )


# --- TRANSFORMER TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.start_transformers",
    bind=True,
    max_retries=5,
)
def start_transformers(self, db_id: int, source_url: str):
    """Starting the element_sim-to-element_dist-transformers plugin"""
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        element_sims_url = extract_output_url(outputs, "relation/element-similarities")
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters
        )

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
            "attributes": task_data.data[f"{WU_PALMER_PLUGIN}_attributes"],
            "transformer": params.transformer.name,
        }
    except Exception as e:
        task_data.add_task_log_entry(
            f"CRASH in Transformers setup:\n{traceback.format_exc()}"
        )
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)
        return

    run_pipeline_step(
        celery_task=self,
        db_id=db_id,
        task_data=task_data,
        plugin_name=TRANSFORMERS_PLUGIN,
        logging_name="Transformers",
        payload=payload,
    )


# --- AGGREGATOR TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.start_aggregator",
    bind=True,
    max_retries=5,
)
def start_aggregator(self, db_id: int, source_url: str):
    """Starting the attribute-distance-aggregator plugin"""
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        element_dists_url = extract_output_url(outputs, "relation/element-distances")
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters
        )

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
    except Exception as e:
        task_data.add_task_log_entry(
            f"CRASH in Aggregator setup:\n{traceback.format_exc()}"
        )
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)
        return

    run_pipeline_step(
        celery_task=self,
        db_id=db_id,
        task_data=task_data,
        plugin_name=AGGREGATOR_PLUGIN,
        logging_name="Aggregator",
        payload=payload,
    )


# --- MDS TASK ---
@CELERY.task(name=f"{Router.instance.identifier}.start_mds", bind=True, max_retries=5)
def start_mds(self, db_id: int, source_url: str):
    """Starting the attribute-distance-mds plugin"""
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        attr_dists_url = extract_output_url(outputs, "relation/attribute-distances")
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters or "{}"
        )

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
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in MDS setup:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)
        return

    run_pipeline_step(
        celery_task=self,
        db_id=db_id,
        task_data=task_data,
        plugin_name=MDS_PLUGIN,
        logging_name="MDS",
        payload=payload,
    )


# --- END OF PIPELINE ---
@CELERY.task(
    name=f"{Router.instance.identifier}.finalize_pipeline", bind=True, max_retries=5
)
def finalize_pipeline(self, db_id: int, source_url: str):
    """Finalizing the pipeline after MDS execution.
    If not finishing with vector concat, writing the final vector zip files to output."""
    TASK_LOGGER.info("Finishing the Pipeline")
    task_data = ProcessingTask.get_by_id(db_id)

    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        final_dists_url = extract_output_url(outputs, "entity/vector")
        current_pipeline_name = task_data.data.get("current_pipeline", "unknown")

        if is_store_mds_output(task_data):
            STORE.persist_task_result(
                db_id,
                open_url(final_dists_url).content,
                f"{current_pipeline_name}_mds_vectors.zip",
                "entity/vector",
                "application/zip",
            )

        # store zip urls
        vector_zip_urls = task_data.data.get("vector_zip_urls", [])
        vector_zip_urls.append(final_dists_url)
        task_data.data["vector_zip_urls"] = vector_zip_urls
        task_data.add_task_log_entry(
            f"Saved vector zip url of {current_pipeline_name} in task_data."
        )
        task_data.save(commit=True)

        # trigger next pipeline
        launch_next_pipeline(task_data)

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        task_data.add_task_log_entry(
            f"Error occurred during finalization step. Attempting to retry:\n{e}"
        )
        task_data.save(commit=True)
        raise self.retry(exc=e, countdown=CELERY_COUNTDOWN)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in final step:\n{traceback.format_exc()}")
        task_data.save(commit=True)
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)


# --- AFTER ALL PIPELINES: Vector concat ---
@CELERY.task(
    name=f"{Router.instance.identifier}.start_vector_concat", bind=True, max_retries=5
)
def start_vector_concat(self, db_id: int):
    """Starting the vector concatenation of the vector urls of each pipeline."""
    task_data = ProcessingTask.get_by_id(db_id)

    try:
        params: InputParameters = InputParametersSchema(unknown=EXCLUDE).loads(
            task_data.parameters or "{}"
        )
        vector_zip_urls = task_data.data.get("vector_zip_urls", [])

        for source_url in vector_zip_urls:
            outputs = requests.get(source_url).json().get("outputs", [])
            final_dists_url = extract_output_url(outputs, "entity/vector")

            if params.include_intermediate_results_in_output:
                prefix = task_data.data.get("current_pipeline", "unknown")
                STORE.persist_task_result(
                    db_id,
                    open_url(final_dists_url).content,
                    f"{prefix}_mds_final_vectors.zip",
                    "entity/vector",
                    "application/zip",
                )

        payload = {
            "urls": vector_zip_urls,
            "output_format": params.output_format,
            "output_suffix": "final_concatenated_vector",
        }
    except Exception as e:
        task_data.add_task_log_entry(
            f"CRASH in Vector concat setup:\n{traceback.format_exc()}"
        )
        task_data.save(commit=True)
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)
        return

    run_pipeline_step(
        celery_task=self,
        db_id=db_id,
        task_data=task_data,
        plugin_name=VECTOR_CONCAT_PLUGIN,
        logging_name="MDS",
        payload=payload,
    )
