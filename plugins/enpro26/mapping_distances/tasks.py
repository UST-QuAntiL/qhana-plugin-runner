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
import math
from io import StringIO
from tempfile import SpooledTemporaryFile
from typing import Callable, Optional, Tuple, List, Dict, Any
from zipfile import ZipFile

from celery.utils.log import get_task_logger

from stable_plugins.file_utils.zip_merger import get_readable_hash

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
    """Extracts the taxanomy name from the attribute metadata"""
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
    if isinstance(val, list):
        return [str(v) for v in val if v]
    if isinstance(val, str):
        if metadata.multiple and metadata.separator:
            return [v.strip() for v in val.split(metadata.separator) if v.strip()]
        return [val.strip()] if val.strip() else []
    return [str(val)]


def calculate_distance(
    elements1: List[str],
    elements2: List[str],
    tax_map: Dict[str, List[float]],
    distance_metric: DistanceMetricEnum,
):
    """
    Calculates the pairwise distances between the two elements based on the metric.\\
    Returns the average of the average of both sums of distances.\\
    Returns 0 if one or both elements are empty.
    """
    # symmetric max-mean inspired distance aggregation for sets of coordinates
    if not elements1 or not elements2:
        return 0.0
    else:

        # Forward pass: distances from e1 to closest e2
        sum1 = 0.0
        for e1 in elements1:
            v1 = tax_map.get(e1, [])
            min_d = min(
                [
                    calculate_vector_distance(v1, tax_map.get(e2, []), distance_metric)
                    for e2 in elements2
                ]
            )
            sum1 += min_d
        avg1 = sum1 / len(elements1)

        # Backward pass: distances from e2 to closest e1
        sum2 = 0.0
        for e2 in elements2:
            v2 = tax_map.get(e2, [])
            min_d = min(
                [
                    calculate_vector_distance(v2, tax_map.get(e1, []), distance_metric)
                    for e1 in elements1
                ]
            )
            sum2 += min_d
        avg2 = sum2 / len(elements2)

        return (avg1 + avg2) / 2.0


def calculate_vector_distance(
    vec1: List[float], vec2: List[float], metric: DistanceMetricEnum
) -> float:
    """Calculates the distance between two coordinate vectors based on the selected metric."""
    if len(vec1) != len(vec2):
        # Fallback padding with zeros if vectors mismatched in dimensionality
        max_len = max(len(vec1), len(vec2))
        vec1 = vec1 + [0.0] * (max_len - len(vec1))
        vec2 = vec2 + [0.0] * (max_len - len(vec2))

    if not vec1:
        return 0.0

    if metric == DistanceMetricEnum.euclidean:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    elif metric == DistanceMetricEnum.manhatten:
        return sum(abs(a - b) for a, b in zip(vec1, vec2))

    elif metric == DistanceMetricEnum.chebyshev:
        return max(abs(a - b) for a, b in zip(vec1, vec2))

    elif metric == DistanceMetricEnum.cosine:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a**2 for a in vec1))
        norm_b = math.sqrt(sum(b**2 for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot_product / (norm_a * norm_b))

    else:
        raise ValueError(f"Unknown distance metric: {metric}")


@CELERY.task(
    name=f"{MappingDistances.instance.identifier}.calculation_task",
    bind=True,
    ignore_result=False,
)
def calculation_task(self, db_id: str) -> str:
    """
    1. Loads the data, similar to the wu_palmer plugin.\\
    2. Extracts the mappings per taxanomy\\
    3. Calculates pairwise distances between all entities per attribute via helper methods\\
    4. Saves the attribute distance data as json files in a zip archive. 
    """
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
            attrib_meta = entities_metadata.get(attribute)
            attribute_distances = []

            tax_name = extract_tax_name(attrib_meta)
            tax_map = taxonomy_mappings.get(tax_name, {})

            # Pairwise distance calculation across all entity instances
            for i in range(len(entities)):
                for j in range(i, len(entities)):

                    ent1 = entities[i]
                    ent2 = entities[j]

                    elements1 = get_element_list(ent1, attribute, attrib_meta)
                    elements2 = get_element_list(ent2, attribute, attrib_meta)
                    dist = calculate_distance(
                        elements1, elements2, tax_map, distance_metric
                    )

                    attribute_distances.append(
                        {
                            "ID": f"{ent1['ID']}__{ent2['ID']}___{attribute}",
                            "entity_1_ID": ent1["ID"],
                            "entity_2_ID": ent2["ID"],
                            "href": "",
                            "distance": dist,
                        }
                    )

            with StringIO() as file:
                save_entities(attribute_distances, file, "application/json")
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
        "custom/attribute-distances",
        "application/zip",
    )

    return "Result stored in file"
