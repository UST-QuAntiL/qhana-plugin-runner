# Copyright 2022 QHAna plugin runner contributors.
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

import os
from tempfile import SpooledTemporaryFile

from typing import Optional, List

from celery.utils.log import get_task_logger

from . import QiskitQKE

from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
    ensure_dict,
)
from qhana_plugin_runner.requests import open_url, retrieve_filename
from qhana_plugin_runner.storage import STORE

import numpy as np
import muid


TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def get_point(ent):
    dimension_keys = list(ent.keys())
    dimension_keys.remove("ID")
    dimension_keys.remove("href")

    dimension_keys.sort()
    point = np.empty(
        (
            len(
                dimension_keys,
            )
        )
    )
    for idx, d in enumerate(dimension_keys):
        point[idx] = ent[d]
    return point


def get_entity_generator(entity_points_url: str):
    """
    Return a generator for the entity points, given an url to them.
    :param entity_points_url: url to the entity points
    """
    file_ = open_url(entity_points_url)
    file_.encoding = "utf-8"
    file_type = file_.headers["Content-Type"]
    entities_generator = load_entities(file_, mimetype=file_type)
    entities_generator = ensure_dict(entities_generator)
    return entities_generator


def get_indices_and_point_arr(entity_points_url: str) -> (dict, List[List[float]]):
    entity_points = list(get_entity_generator(entity_points_url))
    id_to_idx = {}

    idx = 0

    for ent in entity_points:
        if ent["ID"] in id_to_idx:
            raise ValueError("Duplicate ID: ", ent["ID"])

        id_to_idx[ent["ID"]] = idx
        idx += 1

    points_cnt = len(id_to_idx)
    dimensions = len(entity_points[0].keys() - {"ID", "href"})
    points_arr = np.zeros((points_cnt, dimensions))

    for ent in entity_points:
        idx = id_to_idx[ent["ID"]]
        points_arr[idx] = get_point(ent)

    return id_to_idx, points_arr


@CELERY.task(name=f"{QiskitQKE.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new qiskit quantum kernel estimation calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    entity_points_url1 = input_params.entity_points_url1
    entity_points_url2 = input_params.entity_points_url2
    kernel_enum = input_params.kernel
    entanglement_pattern = input_params.entanglement_pattern
    n_qbits = input_params.n_qbits
    paulis = input_params.paulis
    reps = input_params.reps
    shots = input_params.shots
    backend = input_params.backend
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    if ibmq_token == "****":
        TASK_LOGGER.info("Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info("IBMQ_TOKEN environment variable not set")

    # load data from file

    id_to_idx_x, points_arr_x = get_indices_and_point_arr(entity_points_url1)
    id_to_idx_y, points_arr_y = get_indices_and_point_arr(entity_points_url2)

    backend = backend.get_qiskit_backend(ibmq_token, custom_backend)
    backend.shots = shots

    entanglement_pattern = entanglement_pattern.get_pattern()
    paulis = paulis.replace(" ", "").split(",")

    kernel = kernel_enum.get_kernel(backend, n_qbits, paulis, reps, entanglement_pattern)
    # kernel_matrix is size len(points_arr_y) x len(points_arr_x)
    kernel_matrix = kernel.evaluate(x_vec=points_arr_x, y_vec=points_arr_y).T
    TASK_LOGGER.info(f"kernel_matrix.shape = {kernel_matrix.shape}")

    kernel_json = []
    for ent_id_x, idx_x in id_to_idx_x.items():
        for ent_id_y, idx_y in id_to_idx_y.items():
            kernel_json.append(
                {
                    "entity_1_ID": ent_id_y,
                    "entity_2_ID": ent_id_x,
                    "kernel": kernel_matrix[idx_y, idx_x],
                }
            )

    concat_filenames = retrieve_filename(entity_points_url1)
    concat_filenames += retrieve_filename(entity_points_url2)
    filename_hash = get_readable_hash(concat_filenames)

    kernel_name = str(kernel_enum.name).replace("_feature_map", "")

    info_str = f"_qiskit-kernel_{kernel_name}_entanglement_{entanglement_pattern}_{filename_hash}"

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(kernel_json, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"kernel{info_str}.json",
            "custom/kernel-matrix",
            "application/json",
        )

    return "Result stored in file"
