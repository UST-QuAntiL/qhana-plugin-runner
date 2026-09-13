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

import itertools
import json
import math
import sys
from io import StringIO
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Tuple
from zipfile import ZipFile

from celery.utils.log import get_task_logger
from scipy.spatial import distance

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import (
    AttributeMetadata,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.plugin_utils.hashing import get_readable_hash
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import get_mimetype, open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import MappingDistances
from .schemas import (
    NUMERIC_TYPES,
    DistanceMetricEnum,
    InputParameters,
    InputParametersSchema,
)

TASK_LOGGER = get_task_logger(__name__)


def _load_input_parameters(
    db_id: int,
) -> Tuple[str, str, str, list[str], DistanceMetricEnum]:
    """Load and parse the task input parameters from the database."""
    TASK_LOGGER.info(
        f"Starting new Mapping to Distances calculation task with db id '{db_id}'"
    )
    task_data: ProcessingTask | None = ProcessingTask.get_by_id(id_=db_id)

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
    attributes: list[str] = [
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


def _extract_tax_name(attrib_meta: AttributeMetadata) -> str:
    """Extracts the taxonomy name from the attribute metadata.

    Raises a ``ValueError`` if the referenced taxonomy is nested inside a
    subdirectory of the zip (i.e. the name contains a path separator), since
    nested taxonomy zips are not supported.
    """
    tax_name = ""
    if attrib_meta and attrib_meta.ref_target and ":" in attrib_meta.ref_target:
        raw_path = attrib_meta.ref_target.split(":", 1)[1]

        if "/" in raw_path or "\\" in raw_path:
            msg = f"Nested taxonomy zips are not supported: '{raw_path}'"
            TASK_LOGGER.error(msg)
            raise ValueError(msg)

        tax_name = Path(raw_path).stem
    return tax_name


def _get_element_list(
    entity: dict[str, Any], attribute: str, metadata: AttributeMetadata
) -> list[str]:
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
            return [
                stripped for v in val.split(metadata.separator) if (stripped := v.strip())
            ]
        return [val.strip()] if val.strip() else []
    return [str(val)]


def _parse_numeric_value(raw: Any) -> float | None:
    """Parses a raw entity value into a float.

    Returns ``None`` if the value is missing, blank, ``nan``, or cannot be parsed as a
    float.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return None if isinstance(raw, float) and math.isnan(raw) else float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped or stripped.lower() == "nan":
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _parse_numeric_vector(
    entity: dict[str, Any], attribute: str, attrib_meta: AttributeMetadata
) -> list[float] | None:
    """Parses a numeric entity attribute field into a vector of floats.

    A single-valued attribute gives a one-element vector. A multi-valued attribute
    gives one element per value, in the order given by the (already deserialized)
    entity value.

    Returns ``None`` if the entity has no usable value for the attribute (missing,
    empty, or containing an unparsable value).
    """
    val = entity.get(attribute)
    if val is None:
        return None

    if attrib_meta.multiple:
        if isinstance(val, (set, list, dict)) and not val:
            return None
        if isinstance(val, (set, list)):
            raw_values = list(val)
        elif isinstance(val, str):
            raw_values = (
                val.split(attrib_meta.separator) if attrib_meta.separator else [val]
            )
        else:
            raw_values = [val]

        vector = [_parse_numeric_value(v) for v in raw_values]
        if not vector or any(v is None for v in vector):
            return None
        return vector

    value = _parse_numeric_value(val)
    return None if value is None else [value]


def _numeric_element_map(
    entities: list[dict[str, Any]], attribute: str, attrib_meta: AttributeMetadata
) -> dict[str, list[float]]:
    """Builds a map from a stable element key to its parsed numeric vector.

    Mirrors the taxonomy ``tax_map`` structure (element id -> coordinate vector), but
    elements are the distinct parsed values (or value vectors) instead of taxonomy
    element IDs.
    """
    element_map: dict[str, list[float]] = {}
    for entity in entities:
        vector = _parse_numeric_vector(entity, attribute, attrib_meta)
        if vector is None:
            continue
        element_map[json.dumps(vector)] = vector
    return element_map


def _get_numeric_element_list(
    entity: dict[str, Any], attribute: str, attrib_meta: AttributeMetadata
) -> list[str]:
    """Extracts the numeric element key for an entity attribute field.

    Mirrors :func:`_get_element_list`'s signature and return type, but returns at most
    one key, since one entity contributes exactly one numeric element (a single value,
    or a positional value vector for multi-valued attributes).
    """
    vector = _parse_numeric_vector(entity, attribute, attrib_meta)
    return [] if vector is None else [json.dumps(vector)]


def _is_empty_or_nan(vector: list[float]) -> bool:
    """Return ``True`` when a mapping vector is empty or contains any NaN values."""

    if not vector:
        return True
    return any(math.isnan(coordinate) for coordinate in vector)


def _calculate_vector_distance(
    v1: list[float], v2: list[float], metric: DistanceMetricEnum
) -> float:
    """
    Calculates the distance between two coordinate vectors based on the selected metric.

    Raises:
        ValueError: if the mapping vectors do not have the same size.

    Returns:
        float: distance between two coordinate vectors based on the selected metric or
        ``sys.float_info.max`` if the vectors are empty, i.e. no mapping is assigned.
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors do not have the same length")

    if _is_empty_or_nan(v1) or _is_empty_or_nan(v2):
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
    ) = _load_input_parameters(db_id)

    with open_url(entities_metadata_url) as entities_metadata_file:
        entities_metadata_list = list(
            ensure_dict(
                load_entities(
                    entities_metadata_file, get_mimetype(entities_metadata_file)
                )
            )
        )
        entities_metadata = {
            element["ID"]: AttributeMetadata.from_dict(element)
            for element in entities_metadata_list
        }

    with open_url(entities_url) as entities_data:
        mimetype = get_mimetype(entities_data)
        entities = list(
            ensure_dict(load_entities(entities_data, mimetype), entities_metadata)
        )

    taxonomy_mappings: dict[str, dict[str, list[float]]] = {}
    for zipped_file, file_name in get_files_from_zip_url(taxonomies_zip_url, mode="t"):
        tax_json: dict = json.load(zipped_file)
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
            is_numeric = (
                attrib_meta is not None and attrib_meta.description in NUMERIC_TYPES
            )

            if is_numeric:
                tax_map = _numeric_element_map(entities, attribute, attrib_meta)
            else:
                tax_name = _extract_tax_name(attrib_meta)
                tax_map = taxonomy_mappings.get(tax_name, {})

            unique_elements = set()
            for entitiy in entities:
                if is_numeric:
                    entity_names = _get_numeric_element_list(
                        entitiy, attribute, attrib_meta
                    )
                else:
                    entity_names = _get_element_list(entitiy, attribute, attrib_meta)
                unique_elements.update(entity_names)

            elements = sorted(list(unique_elements))
            for e1, e2 in itertools.product(elements, repeat=2):
                v1 = tax_map[e1]
                v2 = tax_map[e2]

                try:
                    dist = _calculate_vector_distance(v1, v2, distance_metric)
                except ValueError as e:
                    TASK_LOGGER.error(
                        f"{e} in mapping_distances plugin task with db_id '{db_id}'"
                    )
                    raise

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
