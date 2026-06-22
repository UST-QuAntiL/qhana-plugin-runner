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
from json import loads
from typing import Callable, Optional, Tuple, List, Dict

from celery.utils.log import get_task_logger

from plugins.enpro26.mapping_distances.schemas import InputParametersSchema
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import (
    AttributeMetadata,
    tuple_deserializer,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    EntityTupleMixin,
    load_entities,
)
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import get_mimetype, open_url

from . import plugin

TASK_LOGGER = get_task_logger(__name__)


def load_input_parameters(db_id: int) -> Tuple[str, str, str, List[str], str]:
    """Load and parse the task input parameters from the database."""
    TASK_LOGGER.info(
        f"Starting new Mapping to Distances calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = InputParametersSchema().loads(task_data.parameters or "{}")

    entities_url: str = params.get("entities_url")
    TASK_LOGGER.info(f"Loaded input parameters from db: entities_url='{entities_url}'")

    entities_metadata_url: str = params.get("entities_metadata_url")
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entities_metadata_url='{entities_metadata_url}'"
    )

    taxonomies_zip_url: str = params.get("taxonomies_zip_url")
    TASK_LOGGER.info(
        f"Loaded input parameters from db: taxonomies_zip_url='{taxonomies_zip_url}'"
    )

    attributes_raw: str = params.get("attributes", "")
    TASK_LOGGER.info(f"Loaded input parameters from db: attributes='{attributes}'")
    attributes: List[str] = [
        attr.strip() for attr in attributes_raw.splitlines() if attr.strip()
    ]

    distance_metric: str = params.get("distance_metric")
    TASK_LOGGER.info(
        f"Loaded parameters: metric='{distance_metric}', attributes={attributes}"
    )
    return (
        entities_url,
        entities_metadata_url,
        taxonomies_zip_url,
        attributes,
        distance_metric,
    )


@CELERY.task(
    name=f"{plugin.MappingDistances.instance.identifier}.calculation_task",
    bind=True,
    ignore_result=False,
)
def calculation_task(self, db_id: str) -> dict:
    (
        entities_url,
        entities_metadata_url,
        taxonomies_zip_url,
        attributes,
        distance_metric,
    ) = load_input_parameters(int(db_id))

    with open_url(entities_metadata_url) as entities_metadata_file:
        entities_metadata_list = list(
            load_entities(entities_metadata_file, get_mimetype(entities_metadata_file))
        )
        entities_metadata = {
            element["ID"]: AttributeMetadata.from_dict(element)
            for element in entities_metadata_list
        }

    deserializer: Optional[Callable[[tuple[str, ...]], tuple[any, ...]]] = None
    entities = []

    with open_url(entities_url) as entities_data:
        mimetype = get_mimetype(entities_data)

        for ent in load_entities(entities_data, mimetype):
            if isinstance(
                ent, EntityTupleMixin
            ):  # Handles NamedTuple serialization format
                if deserializer is None:
                    ent_attributes: tuple[str, ...] = type(ent).entity_attributes
                    ent_tuple = type(ent)
                    deserializer = tuple_deserializer(
                        ent_attributes, entities_metadata, tuple_=ent_tuple._make
                    )
                ent = deserializer(ent)
                entities.append(ent.as_dict())
            else:
                entities.append(ent)

    taxonomies = {}
    for zipped_file, file_name in get_files_from_zip_url(taxonomies_zip_url, mode="t"):
        tax_json: Dict = json.load(zipped_file)
        # Strip '.json' extension to index taxonomies purely by name
        tax_name = file_name[:-5] if file_name.endswith(".json") else file_name
        taxonomies[tax_name] = tax_json

    # TODO Implement distance calculations here

    return {
        "status": "success",
        "message": "Data successfully loaded and ready for calculation",
    }
