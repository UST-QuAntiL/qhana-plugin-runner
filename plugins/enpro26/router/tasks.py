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
from pathlib import PurePath
from typing import Optional
from urllib.parse import urljoin
from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    EntityTupleMixin,
    load_entities,
)
from qhana_plugin_runner.plugin_utils.interop import get_plugin_endpoint
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import get_mimetype, open_url
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result

from . import Router
from .schemas import InputParameters, InputParametersSchema

TASK_LOGGER = get_task_logger(__name__)

# Names of the plugins invoked by the routing pipeline. The runner serves
# plugin metadata at ``/plugins/<name>/`` and redirects a bare name to the
# newest installed version. The route handler turns these into external
# metadata urls (see routes.py), from which the process endpoint is resolved
# through ``get_plugin_endpoint``.
PIPELINE_PLUGINS = {
    "wu_palmer": "wu-palmer",
    "transformer": "element_sim-to-element_dist-transformers",
    "aggregator": "attribute-distance-aggregator",
    "mds": "attribute-distance-mds",
}

# --- HELPER FUNCTIONS ---
def subscribe_to_plugin(task_result_url: str, webhook_url: str):
    """Subscribes the webhook to a target plugin's task result updates."""

    webhook_url = webhook_url.replace("localhost", "127.0.0.1")
    response = requests.get(task_result_url).json()
    subscription_link = next(
        (
            link["href"]
            for link in response.get("links", [])
            if link["type"] == "subscribe"
        ),
        None,
    )
    if not subscription_link:
        raise ValueError("Target plugin does not support subscriptions!")
    requests.post(
        subscription_link,
        json={"command": "subscribe", "event": "status", "webhookHref": webhook_url},
    ).raise_for_status()


def extract_output_url(outputs: list, data_type: str) -> str:
    """Finds the URL of a specific output from a plugin result based on its dataType."""
    for output in outputs:
        if output.get("dataType") == data_type:
            return output["href"]
    raise ValueError(f"Could not find output with dataType {data_type}")


def _plugin_process_url(task_data: ProcessingTask, plugin: str) -> str:
    """Resolve a pipeline plugin's process endpoint from its metadata url.

    The route handler stores the external metadata url per plugin in
    ``data["plugin_urls"]``. ``get_plugin_endpoint`` follows the metadata to
    the plugin's processing endpoint.
    """
    return get_plugin_endpoint(task_data.data["plugin_urls"][plugin])


def _load_task(db_id: int) -> ProcessingTask:
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)
    return task_data


def _taxonomy_ref(metadata: AttributeMetadata) -> Optional[str]:
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


def _load_entity_attributes(entities_url: str) -> set:
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
    TASK_LOGGER.info(f"Starting routing task for db id '{db_id}'")
    if not task_data:
        raise KeyError(f"Could not load task data with id {db_id}")
    params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    TASK_LOGGER.info("Starting Step 1: Wu-Palmer")
    payload = {
        "entitiesUrl": params.entities_url,
        "entitiesMetadataUrl": params.entities_metadata_url,
        "taxonomiesZipUrl": params.taxonomies_zip_url,
        "attributes": task_data.data["wu_palmer_attributes"],
        "root_is_part_of_hierarchy": str(params.root_is_part_of_hierarchy).lower(),
    }

    wu_palmer_url = _plugin_process_url(task_data, "wu_palmer")
    response = requests.post(wu_palmer_url, data=payload, allow_redirects=False)
    task_url = urljoin(wu_palmer_url, response.headers["Location"])

    task_data.data["wu_palmer_url"] = task_url
    task_data.add_task_log_entry("Started Wu-Palmer.")
    task_data.save(commit=True)

    subscribe_to_plugin(task_url, task_data.data["webhook_url"])


# --- Webhook task handles task results ---
@CELERY.task(name=f"{Router.instance.identifier}.handle_webhook_task", bind=True)
def handle_webhook_task(self, db_id: int, source_url: str):
    task_data = ProcessingTask.get_by_id(db_id)

    # Identify which plugin triggered the webhook
    is_wp = source_url == task_data.data.get("wu_palmer_url")
    is_trans = source_url == task_data.data.get("transformers_url")
    is_agg = source_url == task_data.data.get("aggregators_url")
    is_mds = source_url == task_data.data.get("mds_url")

    if not any([is_wp, is_trans, is_agg, is_mds]):
        return "Unrecognized webhook"

    sub_task_result = requests.get(source_url).json()
    status = sub_task_result.get("status", "PENDING")

    if status == "SUCCESS":
        # Launch the dedicated Celery task for whichever step just finished!
        if is_wp:
            process_step_2_transformers.apply_async(args=[db_id, source_url], countdown=4)
        elif is_trans:
            process_step_3_aggregator.apply_async(args=[db_id, source_url], countdown=4)
        elif is_agg:
            process_step_4_mds.apply_async(args=[db_id, source_url], countdown=4)
        elif is_mds:
            finalize_pipeline.apply_async(args=[db_id, source_url], countdown=4)


# --- TRANSFORMER TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.process_step_2_transformers",
    bind=True,
    max_retries=10,
)
def process_step_2_transformers(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 2: Transformer")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        # 1. Persist Transformer result
        outputs = requests.get(source_url).json().get("outputs", [])
        element_sims_url = extract_output_url(outputs, "relation/element-similarities")
        params = InputParametersSchema().loads(task_data.parameters)

        if params.include_intermediate_results_in_output:
            # Wu-Palmer result output
            STORE.persist_task_result(
                db_id,
                open_url(element_sims_url).content,
                "wu_palmer_similarities.zip",
                "relation/element-similarities",
                "application/zip",
            )

        # 2. Launch Transformers
        payload = {
            "similaritiesUrl": element_sims_url,
            "attributes": task_data.data["wu_palmer_attributes"],
            "transformer": params.transformer.name,
        }

        transformer_url = _plugin_process_url(task_data, "transformer")
        response = requests.post(transformer_url, data=payload, allow_redirects=False)
        task_url = urljoin(transformer_url, response.headers["Location"])

        task_data.data["transformers_url"] = task_url
        task_data.add_task_log_entry("Started Transformers.")
        task_data.save(commit=True)
        subscribe_to_plugin(task_url, task_data.data["webhook_url"])

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Step 2:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)


# --- AGGREGATOR TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.process_step_3_aggregator",
    bind=True,
    max_retries=10,
)
def process_step_3_aggregator(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 3: Aggregator")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        element_dists_url = extract_output_url(outputs, "relation/element-distances")
        params = InputParametersSchema().loads(task_data.parameters)

        if params.include_intermediate_results_in_output:
            # Transformer result output
            STORE.persist_task_result(
                db_id,
                open_url(element_dists_url).content,
                "transformer_distances.zip",
                "relation/element-distances",
                "application/zip",
            )

        payload = {
            "entitiesUrl": params.entities_url,
            "elementDistancesUrl": element_dists_url,
        }

        agg_url = _plugin_process_url(task_data, "aggregator")
        response = requests.post(agg_url, data=payload, allow_redirects=False)
        task_url = urljoin(agg_url, response.headers["Location"])

        task_data.data["aggregators_url"] = task_url
        task_data.add_task_log_entry("Started Aggregator.")
        task_data.save(commit=True)
        subscribe_to_plugin(task_url, task_data.data["webhook_url"])

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Step 3:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)


# --- MDS TASK ---
@CELERY.task(
    name=f"{Router.instance.identifier}.process_step_4_mds", bind=True, max_retries=10
)
def process_step_4_mds(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 4: MDS")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        attr_dists_url = extract_output_url(outputs, "relation/attribute-distances")
        params = InputParametersSchema().loads(task_data.parameters)

        if params.include_intermediate_results_in_output:
            # Aggregator result output
            STORE.persist_task_result(
                db_id,
                open_url(attr_dists_url).content,
                "aggregator_distances.zip",
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

        mds_url = _plugin_process_url(task_data, "mds")
        response = requests.post(mds_url, data=payload, allow_redirects=False)

        task_url = urljoin(mds_url, response.headers["Location"])
        task_data.data["mds_url"] = task_url

        task_data.add_task_log_entry("Started MDS.")
        task_data.save(commit=True)

        subscribe_to_plugin(task_url, task_data.data["webhook_url"])
        TASK_LOGGER.info("MDS subscribed")

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Step 4:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)


# --- END OF PIPELINE ---
@CELERY.task(
    name=f"{Router.instance.identifier}.finalize_pipeline", bind=True, max_retries=10
)
def finalize_pipeline(self, db_id: int, source_url: str):
    TASK_LOGGER.info("Starting Step 5: Finishing the Pipeline")
    task_data = ProcessingTask.get_by_id(db_id)
    try:
        outputs = requests.get(source_url).json().get("outputs", [])
        final_dists_url = extract_output_url(outputs, "entity/vector")
        STORE.persist_task_result(
            db_id,
            open_url(final_dists_url).content,
            "mds_final_vectors.json",
            "entity/vector",
            "application/json",
        )

        # TASK COMPLETION
        save_task_result.delay("Pipeline Completed Successfully!", db_id)

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        raise self.retry(exc=e, countdown=3)
    except Exception as e:
        task_data.add_task_log_entry(f"CRASH in Final Step:\n{traceback.format_exc()}")
        save_task_error.delay(failing_task_id=self.request.id, db_id=db_id)
