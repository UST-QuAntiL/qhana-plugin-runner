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
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import get_mimetype, open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import AttributeAggregator
from .schemas import (
    AggregatorsEnum,
    InputParameters,
    InputParametersSchema,
    MissingDataHandling,
)

TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    import muid

    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def _load_entities_as_dicts(entities_url: str) -> List[dict]:
    """Load entities from a URL, resolving attribute metadata for tuple rows."""
    with open_url(entities_url) as entities_data:
        mimetype = get_mimetype(entities_data)
        entities = []
        deserializer: Callable[[tuple[str, ...]], tuple[any, ...]] | None = None
        attribute_metadata: dict[str, AttributeMetadata] = {}

        if "X-Attribute-Metadata" in entities_data.headers:
            attribute_metadata_url = entities_data.headers["X-Attribute-Metadata"]
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
    element_distances_url: str, missing_data_handling: MissingDataHandling
) -> Dict[str, Dict[Tuple[any, any], float]]:
    """Load element distances per attribute and resolve missing (null) distances.

    Returns a mapping from attribute name to a lookup from (source, target)
    element pairs to distance. After missing data handling the lookup contains
    no null distances. With "ignore" the affected pairs are absent instead.
    """
    element_distances = {}

    for file, file_name in get_files_from_zip_url(element_distances_url):
        # removes .json from file name to get the name of the attribute
        attr_name = file_name[:-5]

        loaded_distances = json.load(file)
        distances_without_none: list[float] = [
            dist["distance"] for dist in loaded_distances if dist["distance"] is not None
        ]

        if missing_data_handling == MissingDataHandling.ignore:
            # removes elements with None distance
            loaded_distances = [
                dist for dist in loaded_distances if dist["distance"] is not None
            ]
        elif missing_data_handling == MissingDataHandling.mean:
            if len(distances_without_none) == 0:
                raise ValueError(
                    f"every distance for attribute {attr_name} is None, therefore the mean cannot be calculated"
                )

            mean_distance = sum(distances_without_none) / len(distances_without_none)
            # replaces None distances with the mean distance
            for dist in loaded_distances:
                if dist["distance"] is None:
                    dist["distance"] = mean_distance
        elif missing_data_handling == MissingDataHandling.max:
            if len(distances_without_none) == 0:
                raise ValueError(
                    f"every distance for attribute {attr_name} is None, therefore the maximum cannot be calculated"
                )

            max_distance = max(distances_without_none)
            # replaces None distances with the max distance
            for dist in loaded_distances:
                if dist["distance"] is None:
                    dist["distance"] = max_distance
        else:
            raise NotImplementedError(
                f"Unknown missing_data_handling '{missing_data_handling}'"
            )

        element_distances[attr_name] = {
            (dist["source"], dist["target"]): dist["distance"]
            for dist in loaded_distances
        }

    return element_distances


def _get_dist(elem_dists: Dict[Tuple[any, any], float], val1, val2) -> Optional[float]:
    """Look up the distance of an element pair in both directions."""
    if (val1, val2) in elem_dists:
        return elem_dists[(val1, val2)]
    elif (val2, val1) in elem_dists:
        return elem_dists[(val2, val1)]
    else:
        return None  # handles missing element pairs


def _as_list(values) -> list:
    if isinstance(values, set):
        return list(values)

    if not isinstance(values, list):
        return [values]

    return values


def _aggregate(dist_list: List[float], aggregator: AggregatorsEnum) -> float:
    if aggregator == AggregatorsEnum.mean:
        return sum(dist_list) / len(dist_list)
    elif aggregator == AggregatorsEnum.median:
        dist_list = sorted(dist_list)

        if len(dist_list) % 2 == 0:
            return (
                0.5 * dist_list[len(dist_list) // 2]
                + 0.5 * dist_list[len(dist_list) // 2 - 1]
            )
        else:
            return dist_list[len(dist_list) // 2]
    elif aggregator == AggregatorsEnum.max:
        return max(dist_list)
    elif aggregator == AggregatorsEnum.min:
        return min(dist_list)
    else:
        raise NotImplementedError(f"Unknown aggregator '{aggregator}'")


@CELERY.task(
    name=f"{AttributeAggregator.instance.identifier}.calculation_task", bind=True
)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new attribute distance aggregation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entities_url = input_params.entities_url
    TASK_LOGGER.info(f"Loaded input parameters from db: entities_url='{entities_url}'")
    element_distances_url = input_params.element_distances_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: element_distances_url='{element_distances_url}'"
    )
    attributes: List[str] = input_params.attributes.splitlines()
    TASK_LOGGER.info(f"Loaded input parameters from db: attributes='{attributes}'")
    aggregator = input_params.aggregator
    TASK_LOGGER.info(f"Loaded input parameters from db: aggregator='{aggregator}'")
    missing_data_handling = input_params.missing_data_handling
    TASK_LOGGER.info(
        f"Loaded input parameters from db: missing_data_handling='{missing_data_handling}'"
    )

    # load data from files

    entities = _load_entities_as_dicts(entities_url)
    element_distances = _load_element_distances(
        element_distances_url, missing_data_handling
    )

    # aggregate element distances to attribute distances

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attribute in attributes:
        elem_dists = element_distances[attribute]
        attribute_distances = []

        for i in range(len(entities)):
            for j in range(i, len(entities)):
                ent1 = entities[i]
                ent2 = entities[j]

                values1 = ent1.get(attribute)
                values2 = ent2.get(attribute)

                if values1 is None or values2 is None:
                    dist = None
                else:
                    values1 = _as_list(values1)
                    values2 = _as_list(values2)

                    dist_list = []

                    for val1 in values1:
                        for val2 in values2:
                            elem_dist = _get_dist(elem_dists, val1, val2)

                            if elem_dist is not None:
                                dist_list.append(elem_dist)

                    if len(dist_list) == 0:
                        dist = None
                    else:
                        dist = _aggregate(dist_list, aggregator)

                attribute_distances.append(
                    {
                        "source": ent1["ID"],
                        "target": ent2["ID"],
                        "distance": dist,
                    }
                )

        with StringIO() as file:
            save_entities(attribute_distances, file, "application/json")
            file.seek(0)
            zip_file.writestr(attribute + ".json", file.read())

    zip_file.close()

    concat_filenames = retrieve_filename(entities_url)
    concat_filenames += retrieve_filename(element_distances_url)
    filenames_hash = get_readable_hash(concat_filenames)
    info_str = f"_{aggregator.name}_{filenames_hash}"

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        f"attribute_distances{info_str}.zip",
        "relation/attribute-distances",
        "application/zip",
    )

    return "Result stored in file"
