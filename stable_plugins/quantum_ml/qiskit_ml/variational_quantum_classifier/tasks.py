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

from . import VQC

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
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

import numpy as np

from qiskit.utils import QuantumInstance
from .backend.vqc import QiskitVQC

from sklearn.metrics import accuracy_score

from .backend.visualization import plot_data, plot_confusion_matrix
import muid
import re

TASK_LOGGER = get_task_logger(__name__)


def get_readable_hash(s: str) -> str:
    return muid.pretty(muid.bhash(s.encode("utf-8")), k1=6, k2=5).replace(" ", "-")


def retrieve_filename_from_url(url) -> str:
    """
    Given an url to a file, it returns the name of the file
    :param url: str
    :return: str
    """
    response = open_url(url)
    fname = ""
    if "Content-Disposition" in response.headers.keys():
        fname = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
        if fname[0] == fname[-1] and fname[0] in {'"', "'"}:
            fname = fname[1:-1]
    else:
        fname = url.split("/")[-1]
    response.close()

    # Remove file type endings
    fname = fname.split(".")
    fname = fname[:-1]
    fname = ".".join(fname)

    return fname


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


@CELERY.task(name=f"{VQC.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new quantum variational classifier calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    train_data_url = input_params.train_data_url
    train_labels_url = input_params.train_labels_url
    test_data_url = input_params.test_data_url
    test_labels_url = input_params.test_labels_url
    feature_map_enum = input_params.feature_map
    entanglement_pattern_feature_map_enum = input_params.entanglement_pattern_feature_map
    reps_feature_map = input_params.reps_feature_map
    paulis = input_params.paulis
    vqc_ansatz_enum = input_params.vqc_ansatz
    entanglement_pattern_ansatz_enum = input_params.entanglement_pattern_ansatz
    reps_ansatz = input_params.reps_ansatz
    optimizer_enum = input_params.optimizer
    maxitr = input_params.maxitr
    shots = input_params.shots
    backend = input_params.backend
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend
    resolution = input_params.resolution

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    if ibmq_token == "****":
        TASK_LOGGER.info("Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info("IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info("IBMQ_TOKEN environment variable not set")

    # load data from file
    train_id_to_idx, train_points = get_indices_and_point_arr(train_data_url)
    train_labels, label_to_int, int_to_label = get_label_arr(
        train_labels_url, train_id_to_idx
    )

    test_id_to_idx, test_points = get_indices_and_point_arr(test_data_url)
    test_labels = None
    if test_labels_url != "" and test_labels_url is not None:
        test_labels, label_to_int, int_to_label = get_label_arr(
            test_labels_url,
            test_id_to_idx,
            label_to_int=label_to_int,
            int_to_label=int_to_label,
        )

    # set no. of qubits accordingly
    n_qbits = train_points.shape[1]

    # Prep backend
    backend = backend.get_qiskit_backend(ibmq_token, custom_backend)
    backend = QuantumInstance(backend=backend, shots=shots)

    # Prep feature map
    entanglement_pattern_feature_map = entanglement_pattern_feature_map_enum.get_pattern()
    paulis = paulis.replace(" ", "").split(",")
    feature_map = feature_map_enum.get_featuremap(
        n_qbits, paulis, reps_feature_map, entanglement_pattern_feature_map
    )

    # Prep ansatz
    entanglement_pattern_ansatz = entanglement_pattern_ansatz_enum.get_pattern()
    vqc_ansatz = vqc_ansatz_enum.get_ansatz(
        n_qbits, entanglement_pattern_ansatz, reps_ansatz
    )

    # Prep optimizer
    optimizer = optimizer_enum.get_optimizer(maxitr)

    # VQC
    vqc = QiskitVQC(backend, feature_map, vqc_ansatz, optimizer)
    vqc.fit(train_points, train_labels)
    predictions = [int_to_label[el] for el in vqc.predict(test_points)]
    output_labels = [
        {"ID": ent_id, "href": "", "label": predictions[idx]}
        for ent_id, idx in test_id_to_idx.items()
    ]

    # Prep VQC output
    vqc_output = [
        {
            "num_qubits": n_qbits,
            "feature_map": {
                "name": feature_map_enum.name,
                "entanglement_pattern": entanglement_pattern_feature_map_enum.name,
                "paulis": str(paulis)[1:-1],
                "reps": reps_feature_map,
            },
            "ansatz": {
                "name": vqc_ansatz_enum.name,
                "entanglement_pattern": entanglement_pattern_ansatz_enum.name,
                "reps": reps_ansatz,
            },
            "optimizer": {
                "name": optimizer_enum.name,
                "maxitr": maxitr,
            },
            "weights": vqc.get_weights().tolist(),
        }
    ]

    # Get representative circuit
    representative_circuit = vqc.get_representative_circuit(train_points, train_labels)

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
    fig = plot_data(
        train_points,
        train_id_to_idx,
        train_labels,
        test_points,
        test_id_to_idx,
        predictions,
        resolution=resolution,
        predictor=vqc.predict,
        title=plot_title,
        label_to_int=label_to_int,
    )

    concat_filenames = retrieve_filename_from_url(train_data_url)
    concat_filenames += retrieve_filename_from_url(train_labels_url)
    concat_filenames += retrieve_filename_from_url(test_data_url)
    concat_filenames += retrieve_filename_from_url(test_labels_url)
    filename_hash = get_readable_hash(concat_filenames)

    feature_map_name = str(feature_map_enum.name).replace("_feature_map", "")

    info_str = f"_vqc_feature_map_{feature_map_name}_e_{entanglement_pattern_feature_map}_ansatz_{vqc_ansatz_enum.name}_e_{entanglement_pattern_ansatz}_{filename_hash}"

    # Output
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(output_labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"test_labels{info_str}.json",
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
                f"classification_plot{info_str}.html",
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

    with SpooledTemporaryFile(mode="w") as output:
        save_entities(vqc_output, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            f"vqc_metadata{info_str}.json",
            "vqc-metadata",
            "application/json",
        )

    return "Result stored in file"
