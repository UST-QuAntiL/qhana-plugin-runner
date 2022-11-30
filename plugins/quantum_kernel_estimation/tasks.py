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

from . import QKE

from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities, load_entities, ensure_dict
)
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

import numpy as np


TASK_LOGGER = get_task_logger(__name__)


def get_point(ent):
    dimension_keys = list(ent.keys())
    dimension_keys.remove("ID")
    dimension_keys.remove("href")

    dimension_keys.sort()
    point = np.empty((len(dimension_keys, )))
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
    for ent in entities_generator:
        yield {"ID": ent["ID"], "href": ent.get("href", ""), "point": get_point(ent)}


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
    dimensions = len(entity_points[0]["point"])
    points_arr = np.zeros((points_cnt, dimensions))

    for ent in entity_points:
        idx = id_to_idx[ent["ID"]]
        points_arr[idx] = ent["point"]

    return id_to_idx, points_arr


@CELERY.task(name=f"{QKE.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new quantum kernel estimation calculation task with db id '{db_id}'"
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
    reps = input_params.reps
    shots = input_params.shots
    backend = input_params.backend
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend

    TASK_LOGGER.info(
        f"Loaded input parameters from db: {str(input_params)}"
    )

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

    max_qbits = backend.get_max_num_qbits(ibmq_token, custom_backend)
    if max_qbits is None:
        max_qbits = 6

    backend = backend.get_pennylane_backend(ibmq_token, custom_backend, max_qbits)
    backend.shots = shots

    # entanglement_pattern = entanglement_pattern.get_pattern(n_qbits)
    kernel = kernel_enum.get_kernel(backend, n_qbits, reps, entanglement_pattern)
    # kernel_matrix is size len(points_arr_y) x len(points_arr_x)
    kernel_matrix, representative_circuit = kernel.evaluate(points_arr_x, points_arr_y)

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

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(kernel_json, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "kernel.json",
            "kernel-matrix",
            "application/json",
        )

    with SpooledTemporaryFile(mode="w") as output:
        output.write(representative_circuit)
        STORE.persist_task_result(
            db_id,
            output,
            "representative_circuit.qasm",
            "representative-circuit",
            "application/qasm",
        )

    return "Result stored in file"
