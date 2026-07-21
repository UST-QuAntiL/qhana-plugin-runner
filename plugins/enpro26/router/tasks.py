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
from pathlib import PurePath
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import load_entities
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import get_mimetype, open_url

from . import Router
from .schemas import InputParameters, InputParametersSchema
from .task_helpers import _load_task, _load_entity_attributes, _taxonomy_ref
from .pipeline_tasks import (
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
    "numerical_mapping": "mapping-distances", 
    "transformers": "element_sim-to-element_dist-transformers",
    "aggregator": "attribute-distance-aggregator",
    "mds": "attribute-distance-mds",
}


# --- Step 1: discover taxonomy attributes for the routing step ---
@CELERY.task(name=f"{Router.instance.identifier}.preprocessing_task", bind=True)
def preprocessing_task(self, db_id: int) -> str:
    """Collect the taxonomy attributes presented in the routing step."""
    TASK_LOGGER.info(f"Starting router preprocessing with db id '{db_id}'")
    task_data = _load_task(db_id)

    params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    # The attribute metadata describes the full schema for all entity types. The
    # uploaded entities only contain a subset of those attributes, so collect the
    # attribute names actually present in the entities file.
    entity_attributes = _load_entity_attributes(params.entities_url)

    # Collect the taxonomy file names actually present in the uploaded zip.
    available_taxonomies = {
        PurePath(file_name).name
        for _, file_name in get_files_from_zip_url(params.taxonomies_zip_url, mode="t")
    }

    # Keep attributes that are present in the entities and whose referenced
    # taxonomy exists in the zip.
    taxonomy_attributes = []
    with open_url(params.entities_metadata_url) as response:
        mimetype = get_mimetype(response)
        for element in load_entities(response, mimetype):
            metadata = AttributeMetadata.from_dict(element)
            if metadata.ID not in entity_attributes:
                continue
            ref = _taxonomy_ref(metadata)
            if ref and PurePath(ref).name in available_taxonomies:
                taxonomy_attributes.append(metadata.ID)

    TASK_LOGGER.info(
        f"Found {len(taxonomy_attributes)} taxonomy attribute(s) with a matching "
        f"taxonomy in the zip: {taxonomy_attributes}"
    )
    task_data.data["taxonomy_attributes"] = taxonomy_attributes
    task_data.save(commit=True)

    return f"Found {len(taxonomy_attributes)} taxonomy attribute(s)."


# --- Initial Routing Task Launcher ---
@CELERY.task(name=f"{Router.instance.identifier}.route_task", bind=True)
def route_task(self, db_id: int) -> str:
    task_data = ProcessingTask.get_by_id(id_=db_id)
   
    # Trigger the first pipeline
    launch_next_pipeline(task_data)
    return "Routing task started and pipeline queued."

# --- Webhook task handles task results ---
@CELERY.task(name=f"{Router.instance.identifier}.handle_webhook_task", bind=True)
def handle_webhook_task(self, db_id: int, source_url: str):
    task_data = ProcessingTask.get_by_id(db_id)
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


def handle_wu_palmer_progression(task_data: ProcessingTask, db_id: int, source_url:str):
    """Handle progression of the wu-palmer pipeline"""

    if source_url == task_data.data.get("wu_palmer_url"):
        start_transformers.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("transformers_url"):
        start_aggregator.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("aggregators_url"):
        start_mds.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("mds_url"):
        finalize_pipeline.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)

def handle_mapping_progression(task_data: ProcessingTask, db_id: int, source_url:str):
    """Handle progression of the mapping pipeline"""

    if source_url == task_data.data.get("mapping_url"):
        start_aggregator.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("aggregators_url"):
        start_mds.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)
    elif source_url == task_data.data.get("mds_url"):
        finalize_pipeline.apply_async(args=[db_id, source_url], countdown=CELERY_COUNTDOWN)

