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
import requests
from io import BytesIO
from zipfile import ZipFile
from pathlib import PurePath
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import load_entities
from qhana_plugin_runner.requests import get_mimetype, open_url

from . import Router
from .schemas import InputParameters, InputParametersSchema
from .tasks_helpers import (
    _load_task,
    _load_entity_attributes,
    _taxonomy_ref,
    _calculate_recommendations,
)
from .tasks_pipeline_steps import (
    CELERY_COUNTDOWN,
    launch_next_pipeline,
    start_transformers,
    start_aggregator,
    start_mds,
    finalize_pipeline,
)

TASK_LOGGER = get_task_logger(__name__)

# Names of the plugins invoked by the routing pipeline. The runner serves
# plugin metadata at ``/plugins/<name>/`` and redirects a bare name to the
# newest installed version. The route handler turns these into external
# metadata urls (see routes.py), from which the process endpoint is resolved
# through ``get_plugin_endpoint``.
PIPELINE_PLUGINS = {
    "wu_palmer": "wu-palmer",
    "mapping": "mapping-distances",
    "transformers": "element_sim-to-element_dist-transformers",
    "aggregator": "attribute-distance-aggregator",
    "mds": "attribute-distance-mds",
}


# --- Step 1: discover taxonomy attributes for the routing step ---
@CELERY.task(name=f"{Router.instance.identifier}.preprocessing_task", bind=True)
def preprocessing_task(self, db_id: int) -> str:
    """Collect the taxonomy attributes presented in the routing step view."""
    TASK_LOGGER.info(f"Starting router preprocessing with db id '{db_id}'")
    task_data = _load_task(db_id)

    params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    # The attribute metadata describes the full schema for all entity types. The
    # uploaded entities only contain a subset of those attributes, so collect the
    # attribute names actually present in the entities file.
    entity_attributes = _load_entity_attributes(params.entities_url)

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
            ref = _taxonomy_ref(metadata)
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

                    recommendations[metadata.ID] = _calculate_recommendations(
                        taxonomies_zip, matched_zip_path
                    )

    TASK_LOGGER.info(
        f"Found {len(taxonomy_attributes)} taxonomy attribute(s) with a matching "
        f"taxonomy in the zip: {taxonomy_attributes}"
    )

    task_data.data["taxonomy_attributes"] = taxonomy_attributes
    task_data.data["recommendations"] = recommendations
    task_data.save(commit=True)

    return f"Found {len(taxonomy_attributes)} taxonomy attribute(s)."


# --- Initial Routing Task Launcher ---
@CELERY.task(name=f"{Router.instance.identifier}.start_routing_task", bind=True)
def start_routing_task(self, db_id: int) -> str:
    """Starts the router. Gets the users routing selections for the attributes and puts the corresponding pipelines in a queue."""
    task_data = ProcessingTask.get_by_id(id_=db_id)

    selections = task_data.data.get("routing_selections", {})

    wu_palmer_attributes = [
        attr for attr, option in selections.items() if option == "Wu-Palmer"
    ]
    mapping_attributes = [
        attr for attr, option in selections.items() if option == "Mapping"
    ]
    one_hot_attributes = [
        attr for attr, option in selections.items() if option == "One-Hot"
    ]

    pipeline_queue = []

    if wu_palmer_attributes:
        task_data.add_task_log_entry(
            f"Queued Wu-Palmer pipeline for attributes: {wu_palmer_attributes}"
        )
        task_data.data["wu_palmer_attributes"] = "\n".join(wu_palmer_attributes)
        pipeline_queue.append("wu_palmer")

    if mapping_attributes:
        task_data.add_task_log_entry(
            f"Queued distances mapping pipeline for attributes: {mapping_attributes}"
        )
        task_data.data["mapping_attributes"] = "\n".join(mapping_attributes)
        pipeline_queue.append("mapping")

    if one_hot_attributes:
        task_data.add_task_log_entry(
            f"One-Hot encoding not yet supported. Selected One-Hot for attributes: {one_hot_attributes}"
        )

    none_selected = [attr for attr, option in selections.items() if option == "None"]
    if none_selected:
        task_data.add_task_log_entry(f"None selected attributes skipped: {none_selected}")

    task_data.data["pipeline_queue"] = pipeline_queue
    task_data.data["current_pipeline"] = None
    task_data.save(commit=True)

    # Trigger the first pipeline
    launch_next_pipeline(task_data)
    return "Routing task started and pipeline queued."


# --- Webhook task handles task results ---
@CELERY.task(name=f"{Router.instance.identifier}.handle_webhook_task", bind=True)
def handle_webhook_task(self, db_id: int, source_url: str):
    task_data = ProcessingTask.get_by_id(db_id)

    known_urls = [
        task_data.data.get("wu_palmer_url"),
        task_data.data.get("mapping_url"),
        task_data.data.get("transformers_url"),
        task_data.data.get("aggregators_url"),
        task_data.data.get("mds_url"),
    ]

    if not source_url or source_url not in known_urls:
        return "Unrecognized webhook source"

    sub_task_result = requests.get(source_url).json()

    if sub_task_result.get("status", "PENDING") != "SUCCESS":
        return "Task still pending or failed"

    current_pipeline = task_data.data.get("current_pipeline")

    if current_pipeline == "wu_palmer":
        handle_wu_palmer_progression(task_data, db_id, source_url)

    elif current_pipeline == "mapping":
        pass
        handle_mapping_progression(task_data, db_id, source_url)

    else:
        return "Unrecognized pipeline state"


def handle_wu_palmer_progression(task_data: ProcessingTask, db_id: int, source_url: str):
    """Handle progression of the wu-palmer pipeline"""

    if source_url == task_data.data.get("wu_palmer_url"):
        start_transformers.apply_async(
            args=[db_id, source_url], countdown=CELERY_COUNTDOWN
        )
    elif source_url == task_data.data.get("transformers_url"):
        start_aggregator.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("aggregators_url"):
        start_mds.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("mds_url"):
        finalize_pipeline.apply_async(
            args=[db_id, source_url], countdown=CELERY_COUNTDOWN
        )


def handle_mapping_progression(task_data: ProcessingTask, db_id: int, source_url: str):
    """Handle progression of the mapping pipeline"""

    if source_url == task_data.data.get("mapping_url"):
        start_aggregator.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("aggregators_url"):
        start_mds.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("mds_url"):
        finalize_pipeline.apply_async(
            args=[db_id, source_url], countdown=CELERY_COUNTDOWN
        )
