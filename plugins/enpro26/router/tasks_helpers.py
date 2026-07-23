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
import json
from typing import Optional
from zipfile import ZipFile
from celery.utils.log import get_task_logger

from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    EntityTupleMixin,
    load_entities,
)
from qhana_plugin_runner.plugin_utils.interop import get_plugin_endpoint
from qhana_plugin_runner.requests import get_mimetype, open_url

TASK_LOGGER = get_task_logger(__name__)


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
    Parses the taxonomy JSON to determine the recommended pipeline.
    Returns 'Mapping' if any mapping_raw data is present, otherwise 'Wu-Palmer'.
    """
    # TODO: Refine the recommendation detection (Template also allows for no recommendation)

    try:
        with taxonomies_zip.open(zip_path) as f:
            tax_data = json.load(f)
            # Check if any entity has a non-empty mapping_raw
            has_mapping = any(
                ent.get("mapping_raw", "") != "" for ent in tax_data.get("entities", [])
            )
            return "Mapping" if has_mapping else "Wu-Palmer"
    except Exception as e:
        TASK_LOGGER.warning(f"Could not read mapping for {zip_path}: {e}")
        return "Wu-Palmer"
