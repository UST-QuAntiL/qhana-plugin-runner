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
from itertools import combinations
from pathlib import PurePath
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    ensure_dict,
    load_entities,
    save_entities,
)
from qhana_plugin_runner.plugin_utils.hashing import get_readable_hash
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import get_mimetype, open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import AttributeAggregator

TASK_LOGGER = get_task_logger(__name__)

# Attribute data types (``AttributeMetadata.description``) treated as numeric.
NUMERIC_TYPES = {"number", "integer", "int", "float", "double"}


def _parse_numeric_value(raw: Any) -> Optional[float]:
    """Parse a raw value into a float, or ``None`` if missing/unparseable."""
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
    entity: dict, attribute: str, attrib_meta: AttributeMetadata
) -> Optional[List[float]]:
    """Parse a numeric attribute field into a vector, mirroring mapping_distances.

    Returns ``None`` if the entity has no usable value for the attribute.
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


def _numeric_element_key(vector: List[float]) -> str:
    """Build the same stable string key mapping_distances uses for a vector."""
    return json.dumps(vector)


def _load_entities(entities_url: str) -> Tuple[List[dict], Dict[str, AttributeMetadata]]:
    with open_url(entities_url) as entities_data:
        mimetype = get_mimetype(entities_data)
        attribute_metadata: dict[str, AttributeMetadata] = {}

        attribute_metadata_url = entities_data.headers.get("X-Attribute-Metadata")
        if attribute_metadata_url is None:
            if mimetype == "text/csv":
                raise ValueError(
                    "entities file is text/csv but the X-Attribute-Metadata header is missing"
                )
        else:
            with open_url(attribute_metadata_url) as attribute_metadata_file:
                attribute_metadata = {
                    attr_meta["ID"]: AttributeMetadata.from_dict(attr_meta)
                    for attr_meta in ensure_dict(
                        load_entities(
                            attribute_metadata_file,
                            get_mimetype(attribute_metadata_file),
                        )
                    )
                }

        entities = list(
            ensure_dict(load_entities(entities_data, mimetype), attribute_metadata)
        )

    return entities, attribute_metadata


def _load_element_distances(
    element_distances_url: str,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    element_distances = {}

    for file, file_name in get_files_from_zip_url(element_distances_url):
        attr_name = PurePath(file_name).stem
        loaded_distances = json.load(file)

        for dist in loaded_distances:
            # bool is a subclass of int, so JSON booleans must be rejected explicitly
            if isinstance(dist["distance"], bool) or not isinstance(
                dist["distance"], (int, float)
            ):
                raise ValueError(
                    f"element distance for attribute '{attr_name}' pair "
                    f"({dist['source']}, {dist['target']}) is not a number "
                    f"(got {dist['distance']!r})"
                )

        element_distances[attr_name] = {
            (dist["source"], dist["target"]): dist["distance"]
            for dist in loaded_distances
        }

    return element_distances


def _lookup_element_distance(
    elem_dists: Dict[Tuple[str, str], float], val1, val2, attr_name: str
) -> float:
    if (val1, val2) in elem_dists:
        return elem_dists[(val1, val2)]
    elif (val2, val1) in elem_dists:
        return elem_dists[(val2, val1)]
    else:
        raise ValueError(
            f"element distance for attribute '{attr_name}' pair ({val1}, {val2}) is missing"
        )


def _attribute_values(entity: dict, attribute: str) -> list:
    values = entity.get(attribute)

    if values is None or values == "":
        return []

    if isinstance(values, (list, set)):
        return list(values)

    return [values]


def _attribute_distance(
    ent1: dict,
    ent2: dict,
    attribute: str,
    element_distances: Dict[Tuple[str, str], float],
) -> Optional[float]:
    """Aggregate element distances to an attribute distance with Sym Max Mean.
    adopted from (stable_plugins/classical_ml/data_preparation/sym_max_mean)
    """
    values1 = _attribute_values(ent1, attribute)
    values2 = _attribute_values(ent2, attribute)

    if not values1 or not values2:
        return None

    sum1 = 0.0
    sum2 = 0.0

    for val1 in values1:
        sum1 += min(
            _lookup_element_distance(element_distances, val1, val2, attribute)
            for val2 in values2
        )

    avg1 = sum1 / len(values1)

    for val2 in values2:
        sum2 += min(
            _lookup_element_distance(element_distances, val2, val1, attribute)
            for val1 in values1
        )

    avg2 = sum2 / len(values2)

    return (avg1 + avg2) / 2.0


def _numeric_attribute_distance(
    ent1: dict,
    ent2: dict,
    attribute: str,
    attrib_meta: AttributeMetadata,
    element_distances: Dict[Tuple[str, str], float],
) -> Optional[float]:
    """Look up a numeric attribute's element distance directly, no Sym Max Mean.

    A numeric attribute's value is one element for the whole entity, not a
    set of elements like a categorical multi-valued attribute.
    """
    v1 = _parse_numeric_vector(ent1, attribute, attrib_meta)
    v2 = _parse_numeric_vector(ent2, attribute, attrib_meta)
    if v1 is None or v2 is None:
        return None
    return _lookup_element_distance(
        element_distances, _numeric_element_key(v1), _numeric_element_key(v2), attribute
    )


@CELERY.task(
    name=f"{AttributeAggregator.instance.identifier}.calculation_task", bind=True
)
def calculation_task(self, db_id: int) -> str:
    TASK_LOGGER.info(
        f"Starting new attribute distance aggregation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params = json.loads(task_data.parameters)
    entities_url = params["entitiesUrl"]
    element_distances_url = params["elementDistancesUrl"]

    entities, attribute_metadata = _load_entities(entities_url)
    element_distances_by_attributes = _load_element_distances(element_distances_url)

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attribute, element_distances in element_distances_by_attributes.items():
        attrib_meta = attribute_metadata.get(attribute)
        is_numeric = attrib_meta is not None and attrib_meta.description in NUMERIC_TYPES

        attribute_distances = [
            {
                "source": ent1["ID"],
                "target": ent2["ID"],
                "distance": (
                    _numeric_attribute_distance(
                        ent1, ent2, attribute, attrib_meta, element_distances
                    )
                    if is_numeric
                    else _attribute_distance(ent1, ent2, attribute, element_distances)
                ),
            }
            for ent1, ent2 in combinations(entities, 2)
        ]

        with StringIO() as file:
            save_entities(attribute_distances, file, "application/json")
            file.seek(0)
            zip_file.writestr(attribute + ".json", file.read())

    zip_file.close()

    concat_filenames = retrieve_filename(entities_url)
    concat_filenames += retrieve_filename(element_distances_url)
    filenames_hash = get_readable_hash(concat_filenames)
    info_str = f"_{filenames_hash}"

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        f"attribute_distances{info_str}.zip",
        "relation/attribute-distances",
        "application/zip",
    )

    return "Result stored in file"
