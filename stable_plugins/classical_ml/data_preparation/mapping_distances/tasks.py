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
from io import StringIO
import sys
from tempfile import SpooledTemporaryFile
from typing import Callable, Optional, Tuple, List, Dict, Any
from zipfile import ZipFile

from celery.utils.log import get_task_logger
from scipy.spatial import distance

from qhana_plugin_runner.plugin_utils.hashing import get_readable_hash

from .schemas import InputParametersSchema, InputParameters, DistanceMetricEnum
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import (
    AttributeMetadata,
    tuple_deserializer,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    EntityTupleMixin,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import get_mimetype, open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import MappingDistances

TASK_LOGGER = get_task_logger(__name__)


def load_input_parameters(
    db_id: int,
) -> Tuple[str, str, str, List[str], DistanceMetricEnum]:
    """Load and parse the task input parameters from the database."""
    TASK_LOGGER.info(
        f"Starting new Mapping to Distances calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: InputParameters = InputParametersSchema().loads(task_data.parameters or "{}")

    entities_url: str = params.entities_url
    TASK_LOGGER.info(f"Loaded input parameters from db: entities_url='{entities_url}'")

    entities_metadata_url: str = params.entities_metadata_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entities_metadata_url='{entities_metadata_url}'"
    )

    taxonomies_zip_url: str = params.taxonomies_zip_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: taxonomies_zip_url='{taxonomies_zip_url}'"
    )

    attributes_raw: str = params.attributes
    TASK_LOGGER.info(f"Loaded input parameters from db: attributes='{attributes_raw}'")
    attributes: List[str] = [
        attr.strip() for attr in attributes_raw.splitlines() if attr.strip()
    ]

    distance_metric: DistanceMetricEnum = params.distance_metric
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


def extract_tax_name(attrib_meta: AttributeMetadata) -> str:
    """Extracts the taxonomy name from the attribute metadata"""
    tax_name = ""
    if (
        attrib_meta
        and attrib_meta.ref_target
        and "taxonomies.zip:" in attrib_meta.ref_target
    ):
        tax_name = attrib_meta.ref_target.split("taxonomies.zip:")[1]
        if tax_name.endswith(".json"):
            tax_name = tax_name[:-5]
    return tax_name


def get_element_list(
    entity: Dict[str, Any], attribute: str, metadata: AttributeMetadata
) -> List[str]:
    """Extracts taxonomy element IDs from an entity attribute field."""
    val = entity.get(attribute)
    if val is None:
        return []
    if isinstance(val, (set, list, dict)) and not val:
        return []
    if isinstance(val, (set, list)):
        return [str(v) for v in val if v]
    if isinstance(val, str):
        if metadata.multiple and metadata.separator:
            return [v.strip() for v in val.split(metadata.separator) if v.strip()]
        return [val.strip()] if val.strip() else []
    return [str(val)]


def calculate_vector_distance(
    v1: List[float], v2: List[float], metric: DistanceMetricEnum, db_id: int
) -> float:
    """
    Calculates the distance between two coordinate vectors based on the selected metric.

    Returns ``sys.float_info.max`` if the vectors are empty, i.e. no mapping is assigned.

    Throws a ``ValueError`` if the mapping vectors do not have the same size.
    """
    if len(v1) != len(v2):
        msg = f"Vectors do not have the same length in mapping_distances plugin task with db_id '{db_id}'"
        TASK_LOGGER.error(msg)
        raise ValueError(msg)

    if not v1 or not v2:
        return sys.float_info.max

    if metric == DistanceMetricEnum.euclidean:
        return distance.euclidean(v1, v2)
    elif metric == DistanceMetricEnum.manhatten:
        return distance.cityblock(v1, v2)
    elif metric == DistanceMetricEnum.chebyshev:
        return distance.chebyshev(v1, v2)
    elif metric == DistanceMetricEnum.cosine:
        if all(v == 0 for v in v1) or all(v == 0 for v in v2):
            # If one vector is a zero-vector, the cosine similarity/distance can't be calculated.
            # 2 is the maximum value, that the cosine distance outputs.
            return 2
        return distance.cosine(v1, v2)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


@CELERY.task(
    name=f"{MappingDistances.instance.identifier}.calculation_task",
    bind=True,
    ignore_result=False,
)
def calculation_task(self, db_id: int) -> str:
    """
    1. Loads the data, similar to the wu_palmer plugin, see :func:`load_input_parameters`
    2. Extracts the mappings per taxonomy
    3. Calculates pairwise distances between all active unique elements within each attribute, see :func:`calculate_vector_distance`
    4. Saves the element distance data as json files in a zip archive.
    """
    (
        entities_url,
        entities_metadata_url,
        taxonomies_zip_url,
        attributes,
        distance_metric,
    ) = load_input_parameters(db_id)

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
            if isinstance(ent, EntityTupleMixin):
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

    taxonomy_mappings: Dict[str, Dict[str, List[float]]] = {}
    for zipped_file, file_name in get_files_from_zip_url(taxonomies_zip_url, mode="t"):
        tax_json: Dict = json.load(zipped_file)
        tax_name = file_name[:-5] if file_name.endswith(".json") else file_name

        # Build map: item_id -> numerical vector coordinates
        item_map = {}
        for ent_node in tax_json.get("entities", []):
            mapping_vector = ent_node.get("mapping", [])
            item_map[ent_node["ID"]] = [float(x) for x in mapping_vector]

        taxonomy_mappings[tax_name] = item_map

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    with ZipFile(tmp_zip_file, "w") as zip_file:
        for attribute in attributes:
            element_distances = []

            attrib_meta = entities_metadata.get(attribute)
            tax_name = extract_tax_name(attrib_meta)
            tax_map = taxonomy_mappings.get(tax_name, {})

            unique_elements = set()
            for entitiy in entities:
                entity_names = get_element_list(entitiy, attribute, attrib_meta)
                unique_elements.update(entity_names)

            elements = sorted(list(unique_elements))
            for e1 in elements:
                for e2 in elements:
                    v1 = tax_map[e1]
                    v2 = tax_map[e2]

                    dist = calculate_vector_distance(v1, v2, distance_metric, db_id)

                    element_distances.append(
                        {
                            "source": e1,
                            "target": e2,
                            "distance": dist,
                        }
                    )

            with StringIO() as file:
                save_entities(element_distances, file, "application/json")
                file.seek(0)
                zip_file.writestr(f"{attribute}.json", file.read())

    zip_file.close()

    concat_filenames = retrieve_filename(entities_url)
    concat_filenames += retrieve_filename(entities_metadata_url)
    concat_filenames += retrieve_filename(taxonomies_zip_url)
    filenames_hash = get_readable_hash(concat_filenames)

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        f"mapping_distances_with_metric_{distance_metric.name}_from_{filenames_hash}.zip",
        "relation/element-distances",
        "application/zip",
    )

    return "Result stored in file"
