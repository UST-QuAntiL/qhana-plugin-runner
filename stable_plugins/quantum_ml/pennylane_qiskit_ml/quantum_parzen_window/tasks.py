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

from typing import Optional, List

from celery.utils.log import get_task_logger

from . import QParzenWindow
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
from sklearn.metrics import accuracy_score

from .backend.visualize import plot_data, plot_confusion_matrix
import muid

TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def get_point(ent: dict) -> np.ndarray:
    dimension_keys = [k for k in ent.keys() if k not in ("ID", "href")]
    dimension_keys.sort()
    point = np.empty(len(dimension_keys))
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


def get_label_generator(entity_labels_url: str):
    """
    Return a generator for the entity labels, given an url to them.
    :param entity_labels_url: url to the entity labels
    """
    file_ = open_url(entity_labels_url)
    file_.encoding = "utf-8"
    file_type = file_.headers["Content-Type"]
    entities_generator = load_entities(file_, mimetype=file_type)
    entities_generator = ensure_dict(entities_generator)
    for ent in entities_generator:
        yield {"ID": ent["ID"], "href": ent.get("href", ""), "label": ent["label"]}


def get_label_arr(
    entity_labels_url: str, id_to_idx: dict, label_to_int=None, int_to_label=None
) -> (dict, List[List[float]]):
    entity_labels = list(get_label_generator(entity_labels_url))

    # Initialise label array
    labels = np.zeros(len(id_to_idx.keys()), dtype=int)

    if label_to_int is None:
        label_to_int = dict()
    if int_to_label is None:
        int_to_label = list()
    for ent in entity_labels:
        label = ent["label"]
        label_str = str(label)
        if label_str not in label_to_int:
            label_to_int[label_str] = len(int_to_label)
            int_to_label.append(label)
        labels[id_to_idx[ent["ID"]]] = label_to_int[label_str]

    return labels, label_to_int, int_to_label


@CELERY.task(name=f"{QParzenWindow.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(f"Starting new quantum k ne calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # Load input parameters
    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    # Assign input parameters to variables
    train_points_url = input_params.train_points_url
    train_label_points_url = input_params.train_label_points_url
    test_points_url = input_params.test_points_url
    test_label_points_url = input_params.test_label_points_url
    window_size = input_params.window_size
    variant = input_params.variant
    minimize_qubit_count = input_params.minimize_qubit_count
    backend = input_params.backend
    shots = input_params.shots
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend
    visualize = input_params.visualize

    # Log information about the input parameters
    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")
    if ibmq_token == "****":
        TASK_LOGGER.info("Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info("IBMQ_TOKEN environment variable not set")

    # Load in data
    train_id_to_idx, train_data = get_indices_and_point_arr(train_points_url)
    train_labels, label_to_int, int_to_label = get_label_arr(
        train_label_points_url, train_id_to_idx
    )

    test_id_to_idx, test_data = get_indices_and_point_arr(test_points_url)
    test_labels = None
    if test_label_points_url != "" and test_label_points_url is not None:
        test_labels, label_to_int, int_to_label = get_label_arr(
            test_label_points_url,
            test_id_to_idx,
            label_to_int=label_to_int,
            int_to_label=int_to_label,
        )

    # Retrieve max qubit count
    max_qbits = backend.get_max_num_qbits(ibmq_token, custom_backend)
    if max_qbits is None:
        max_qbits = 20

    # Get parzen window
    parzen_window, num_qbits = variant.get_parzen_window(
        train_data,
        train_labels,
        window_size,
        max_qbits,
        use_access_wires=(not minimize_qubit_count),
    )

    # Set backend
    backend = backend.get_pennylane_backend(ibmq_token, custom_backend, num_qbits)
    backend.shots = shots
    parzen_window.set_quantum_backend(backend)

    # Label test data
    predictions = [int_to_label[el] for el in parzen_window.label_points(test_data)]

    # Prepare labels to be saved
    output_labels = []

    for ent_id, idx in test_id_to_idx.items():
        output_labels.append({"ID": ent_id, "href": "", "label": predictions[idx]})

    # Get representative circuit
    representative_circuit = parzen_window.get_representative_circuit(test_data)

    # Correct train labels
    train_labels = [int_to_label[el] for el in train_labels]

    # Plot title + confusion matrix
    plot_title = "Classification"
    conf_matrix = None
    if test_labels is not None:
        test_labels = [int_to_label[el] for el in test_labels]

        # Compute accuracy on test data
        test_accuracy = accuracy_score(test_labels, predictions)
        plot_title += f": accuracy on test data={test_accuracy}"

        # Create confusion matrix plot
        conf_matrix = plot_confusion_matrix(
            test_labels, predictions, list(set(train_labels))
        )

    # Create plot
    fig = None
    if visualize:
        fig = plot_data(
            train_data,
            train_id_to_idx,
            train_labels,
            test_data,
            test_id_to_idx,
            predictions,
            title=plot_title,
            label_to_int=label_to_int,
        )

    concat_filenames = retrieve_filename(train_points_url)
    concat_filenames += retrieve_filename(train_label_points_url)
    concat_filenames += retrieve_filename(test_points_url)
    concat_filenames += retrieve_filename(test_label_points_url)
    filename_hash = get_readable_hash(concat_filenames)

    variant_name = str(variant.name).replace("window", "").strip("_")

    info_str = (
        f"_q-parzen-window_variant_{variant_name}_window_{window_size}_{filename_hash}"
    )

    # Output the data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(output_labels, output, "application/json")
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
                f"plot{info_str}.html",
                "plot",
                "text/html",
            )

    if conf_matrix is not None:
        with SpooledTemporaryFile(mode="wt") as output:
            html = conf_matrix.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                f"confusion_matrix{info_str}.html",
                "plot",
                "text/html",
            )

    with SpooledTemporaryFile(mode="w") as output:
        output.write(representative_circuit)
        STORE.persist_task_result(
            db_id,
            output,
            f"representative_circuit{info_str}.qasm",
            "representative-circuit",
            "application/qasm",
        )

    return "Result stored in file"
