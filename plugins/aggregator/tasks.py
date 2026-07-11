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
from itertools import combinations
from pathlib import PurePath
from tempfile import SpooledTemporaryFile
from typing import Callable, Dict, List, Optional, Tuple
from zipfile import ZipFile

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.attributes import (
    AttributeMetadata,
    tuple_deserializer,
)
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    EntityTupleMixin,
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


def _load_entities(entities_url: str) -> List[dict]:
    with open_url(entities_url) as entities_data:
        mimetype = get_mimetype(entities_data)
        entities = []
        deserializer: Callable[[tuple[str, ...]], tuple[any, ...]] | None = None
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

        for ent in load_entities(entities_data, mimetype):
            if isinstance(ent, EntityTupleMixin):  # is NamedTuple
                if deserializer is None:
                    deserializer = tuple_deserializer(
                        type(ent).entity_attributes,
                        attribute_metadata,
                        tuple_=type(ent)._make,
                    )

                ent = deserializer(ent)
                entities.append(ent.as_dict())
            else:
                entities.append(ent)

    return entities


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

    entities = _load_entities(entities_url)
    element_distances_by_attributes = _load_element_distances(element_distances_url)

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attribute, element_distances in element_distances_by_attributes.items():
        attribute_distances = [
            {
                "source": ent1["ID"],
                "target": ent2["ID"],
                "distance": _attribute_distance(ent1, ent2, attribute, element_distances),
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
