import os
from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

from . import QKNN
from .schemas import (
    InputParameters,
    InputParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities
from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.storage import STORE

import numpy as np

from .backend.visualize import plot_data

TASK_LOGGER = get_task_logger(__name__)


def get_point(ent):
    dimension_keys = list(ent.keys())
    dimension_keys.remove("ID")
    dimension_keys.remove("href")

    dimension_keys.sort()
    point = np.empty(len(dimension_keys))
    for idx, d in enumerate(dimension_keys):
        point[idx] = ent[d]
    return point


def load_entity_points_from_url(entity_points_url: str):
    """
    Loads in entity points, given their url.
    :param entity_points_url: url to the entity points
    """

    entity_points = open_url(entity_points_url).json()
    id_to_idx = {}

    idx = 0

    for ent in entity_points:
        if ent["ID"] in id_to_idx:
            raise ValueError("Duplicate ID: ", ent["ID"])

        id_to_idx[ent["ID"]] = idx
        idx += 1

    points_cnt = len(id_to_idx)
    dimensions = len(entity_points[0].keys()-{"ID", "href"})
    points_arr = np.zeros((points_cnt, dimensions))

    for ent in entity_points:
        idx = id_to_idx[ent["ID"]]
        points_arr[idx] = get_point(ent)

    return points_arr, id_to_idx


def load_labels_from_url(labels_url: str, id_to_idx: dict):
    # load data from file

    labels = open_url(labels_url).json()

    num_labels = len(id_to_idx)
    label_arr = np.empty((num_labels,), dtype=int)

    for label in labels:
        idx = id_to_idx[label["ID"]]
        label_arr[idx] = label["label"]
    return label_arr


@CELERY.task(name=f"{QKNN.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    # get parameters

    TASK_LOGGER.info(
        f"Starting new quantum k nearest neighbours calculation task with db id '{db_id}'"
    )
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    # Load input parameters
    input_params: InputParameters = InputParametersSchema().loads(task_data.parameters)

    # Assign input parameters to variables
    train_points_url = input_params.train_points_url
    label_points_url = input_params.label_points_url
    test_points_url = input_params.test_points_url
    k = input_params.k
    variant = input_params.variant
    exp_itr = input_params.exp_itr
    minimize_qubit_count = input_params.minimize_qubit_count
    backend = input_params.backend
    shots = input_params.shots
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend
    resolution = input_params.resolution

    # Log information about the input parameters
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


    # Load in data
    train_data, train_id_to_idx = load_entity_points_from_url(train_points_url)
    train_labels = load_labels_from_url(label_points_url, train_id_to_idx)
    test_data, test_id_to_idx = load_entity_points_from_url(test_points_url)

    # Retrieve max qubit count
    max_qbits = backend.get_max_num_qbits(ibmq_token, custom_backend)
    if max_qbits is None:
        max_qbits = 20

    # Get QkNN
    qknn, num_qbits = variant.get_qknn_and_total_wires(
        train_data, train_labels, k, max_qbits,
        exp_itr=exp_itr,
        use_access_wires=(not minimize_qubit_count)
    )

    # Set backend
    backend = backend.get_pennylane_backend(ibmq_token, custom_backend, num_qbits)
    backend.shots = shots
    qknn.set_quantum_backend(backend)

    print(f"backend.num_qubits = {backend.num_wires}")

    # Label test data
    test_labels = qknn.label_points(test_data)

    # Prepare labels to be saved
    output_labels = []

    for ent_id, idx in test_id_to_idx.items():
        output_labels.append({"ID": ent_id, "href": "", "label": int(test_labels[idx])})

    # Get representative circuit
    representative_circuit = qknn.get_representative_circuit(test_data)

    # Create plot
    fig = plot_data(train_data, train_id_to_idx, train_labels, test_data, test_id_to_idx, test_labels,
                    resolution=resolution, predictor=qknn.label_points)

    # Output the data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(output_labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "labels.json",
            "labels",
            "application/json",
        )

    if fig is not None:
        with SpooledTemporaryFile(mode="wt") as output:
            html = fig.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                "plot.html",
                "plot",
                "text/html",
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
