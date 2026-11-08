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
from pathlib import PurePath
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

from celery.utils.log import get_task_logger

from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.plugin_utils.hashing import get_readable_hash
from qhana_plugin_runner.plugin_utils.zip_utils import get_files_from_zip_url
from qhana_plugin_runner.requests import retrieve_filename
from qhana_plugin_runner.storage import STORE

from . import AttributeMds
from .schemas import (
    InputParameters,
    InputParametersSchema,
    MetricEnum,
    MissingDataHandling,
)

TASK_LOGGER = get_task_logger(__name__)

NONMETRIC_ZERO_FACTOR = 0.000001


def _load_attribute_distances(attribute_distances_url: str) -> Dict[str, List[dict]]:
    attribute_distances = {}

    for file, file_name in get_files_from_zip_url(attribute_distances_url):
        attr_name = PurePath(file_name).stem
        attribute_distances[attr_name] = json.load(file)

    return attribute_distances


def _replace_missing_distances(
    distances: List[dict], missing_data_handling: MissingDataHandling, attr_name: str
):
    known_distances = [
        dist["distance"] for dist in distances if dist["distance"] is not None
    ]

    if len(known_distances) == len(distances):
        return

    if not known_distances:
        raise ValueError(
            f"every distance for attribute '{attr_name}' is missing, "
            "therefore no replacement can be calculated"
        )

    if missing_data_handling == MissingDataHandling.mean:
        replacement = sum(known_distances) / len(known_distances)
    else:
        replacement = max(known_distances)

    for dist in distances:
        if dist["distance"] is None:
            dist["distance"] = replacement


def _build_distance_matrix(
    distances: List[dict], attr_name: str
) -> Tuple[Dict[str, int], Any]:
    import numpy as np

    id_to_idx: Dict[str, int] = {}

    for dist in distances:
        for ent_id in (dist["source"], dist["target"]):
            if ent_id not in id_to_idx:
                id_to_idx[ent_id] = len(id_to_idx)

    distance_matrix = np.full((len(id_to_idx), len(id_to_idx)), np.nan)
    np.fill_diagonal(distance_matrix, 0)

    for dist in distances:
        idx_1 = id_to_idx[dist["source"]]
        idx_2 = id_to_idx[dist["target"]]
        distance_matrix[idx_1, idx_2] = dist["distance"]
        distance_matrix[idx_2, idx_1] = dist["distance"]

    missing_indices = np.argwhere(np.isnan(distance_matrix))
    if missing_indices.size:
        idx_to_id = {idx: ent_id for ent_id, idx in id_to_idx.items()}
        missing_pairs = sorted(
            {tuple(sorted((idx_to_id[i], idx_to_id[j]))) for i, j in missing_indices}
        )
        raise ValueError(
            f"attribute '{attr_name}' has no distance entry for entity pairs: "
            + ", ".join(f"({source}, {target})" for source, target in missing_pairs)
        )

    return id_to_idx, distance_matrix


def _replace_zero_distances(distance_matrix):
    """Replace off-diagonal zeros with a value smaller than the smallest
    distance because nonmetric MDS in scikit-learn treats dissimilarities of 0
    as missing values.
    https://scikit-learn.org/1.1/modules/generated/sklearn.manifold.MDS.html?highlight=mds#sklearn.manifold.MDS
    """
    import numpy as np

    zero_mask = distance_matrix == 0
    np.fill_diagonal(zero_mask, False)
    if not zero_mask.any():
        return

    positive_distances = distance_matrix[distance_matrix > 0]
    if positive_distances.size:
        replacement = positive_distances.min() * NONMETRIC_ZERO_FACTOR
        if replacement == 0:
            # the product underflowed to 0, use the smallest positive float
            replacement = np.nextafter(0, positive_distances.min())
    else:
        replacement = NONMETRIC_ZERO_FACTOR
    distance_matrix[zero_mask] = replacement


def _get_dim_attributes(dim: int) -> List[str]:
    zero_padding = len(str(dim - 1))
    return [f"dim{d:0{zero_padding}}" for d in range(dim)]


@CELERY.task(name=f"{AttributeMds.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    from sklearn import manifold

    TASK_LOGGER.info(f"Starting new attribute MDS calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    params: InputParameters = InputParametersSchema().loads(task_data.parameters)
    attribute_distances = _load_attribute_distances(params.attribute_distances_url)

    if not attribute_distances:
        raise ValueError("the attribute distances zip file contains no distance files")

    metric_name = str(params.metric.name).removesuffix("_mds")
    filename = retrieve_filename(params.attribute_distances_url)
    filenames_hash = get_readable_hash(filename)
    info_str = (
        f"_mds_dim_{params.dimensions}_{metric_name}"
        f"_{params.missing_data_handling.name}_from_{filename}_{filenames_hash}"
    )

    tmp_zip_file = SpooledTemporaryFile(mode="wb")
    zip_file = ZipFile(tmp_zip_file, "w")

    for attr_name, distances in attribute_distances.items():
        _replace_missing_distances(distances, params.missing_data_handling, attr_name)
        id_to_idx, distance_matrix = _build_distance_matrix(distances, attr_name)

        if params.metric == MetricEnum.nonmetric_mds:
            _replace_zero_distances(distance_matrix)

        mds = manifold.MDS(
            params.dimensions,
            metric=params.metric == MetricEnum.metric_mds,
            n_init=params.n_init,
            max_iter=params.max_iter,
            dissimilarity="precomputed",
        )
        transformed = mds.fit_transform(distance_matrix)

        entity_points = []
        dim_attributes = _get_dim_attributes(params.dimensions)

        for ent_id, idx in id_to_idx.items():
            new_entity_point = {"ID": ent_id, "href": ""}
            new_entity_point.update(
                {d: x for d, x in zip(dim_attributes, transformed[idx])}
            )
            entity_points.append(new_entity_point)

        with StringIO() as file:
            save_entities(entity_points, file, "application/json")
            file.seek(0)
            zip_file.writestr(attr_name + ".json", file.read())

    zip_file.close()

    STORE.persist_task_result(
        db_id,
        tmp_zip_file,
        f"entity_points{info_str}.zip",
        "entity/vector",
        "application/zip",
    )

    return "Result stored in file"
