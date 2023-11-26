# Copyright 2023 QHAna plugin runner contributors.
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

from typing import Optional

from celery.utils.log import get_task_logger

from . import MaxCut

from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.api.util import retrieve_filename

import numpy as np

from .backend.load_utils import load_matrix_url
from .backend.max_cut_clustering import MaxCutClustering
from .backend.visualize import plot_graph


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{MaxCut.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new max cut calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    adjacency_matrix_url = input_params.adjacency_matrix_url
    max_cut_enum = input_params.max_cut_enum
    num_clusters = input_params.num_clusters
    optimizer = input_params.optimizer
    max_trials = input_params.max_trials
    reps = input_params.reps
    entanglement_pattern_enum = input_params.entanglement_pattern_enum
    backend = input_params.backend
    shots = input_params.shots
    ibmq_custom_backend = input_params.ibmq_custom_backend
    ibmq_token = input_params.ibmq_token
    visualize = input_params.visualize

    if ibmq_token == "****":
        TASK_LOGGER.info("Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info("IBMQ_TOKEN environment variable not set")

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    # Load data
    id_list1, id_list2, adjacency_matrix = load_matrix_url(adjacency_matrix_url)
    adjacency_matrix = np.array(adjacency_matrix)

    # Prepare quantum parameters
    quantum_parameters = dict(
        backend=backend.get_qiskit_backend(ibmq_token, ibmq_custom_backend, shots),
        optimizer=optimizer.get_optimizer(max_trials),
        reps=reps,
        entanglement=entanglement_pattern_enum.get_entanglement_pattern(),
    )

    # Cluster data
    max_cut_solver = max_cut_enum.get_solver(**quantum_parameters)
    max_cut_cluster = MaxCutClustering(max_cut_solver, num_clusters)
    predictions = max_cut_cluster.create_cluster(adjacency_matrix)

    labels = [
        {"ID": _id, "href": "", "label": int(_label)}
        for _id, _label in zip(id_list1, predictions)
    ]

    fig = None
    if visualize:
        fig = plot_graph(
            adjacency_matrix,
            predictions,
            title=f"MaxCut Clusters",
        )

    filename = retrieve_filename(adjacency_matrix_url)
    total_num_clusters = 2**num_clusters

    info_str = f"_max-cut_solver_{max_cut_enum.name}_clusters_{total_num_clusters}_from_{filename}"

    # Output data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"labels{info_str}.json",
            "entity/label",
            "application/json",
        )

    if fig is not None:
        with SpooledTemporaryFile(mode="wt") as output:
            html = fig.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                f"cluster_plot{info_str}.html",
                "plot",
                "text/html",
            )

    return "Result stored in file"
