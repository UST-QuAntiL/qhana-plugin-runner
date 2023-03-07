# hybrid quantum neural network implemented based on:
# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
# https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html

import os
from tempfile import SpooledTemporaryFile

from typing import Optional, List

from celery.utils.log import get_task_logger

from plugins.qnn import QNN
from plugins.qnn.schemas import (
    InputParameters,
    QNNParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url

from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
    load_entities,
    ensure_dict,
)

import numpy as np

# Timing tool
import time

# Scikit-learn tools
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from sklearn.metrics import accuracy_score

from .backend.datasets import OneHotDataset, digits2position
from .backend.train_and_test import train
from .backend.visualize import plot_data, plot_confusion_matrix

# Plot to html
import base64
from io import BytesIO

TASK_LOGGER = get_task_logger(__name__)


# ------------------------------
#       dataset generation
# ------------------------------
def twospirals(n_points, noise=0.7, turns=1.52):
    """Returns the two spirals dataset."""
    n = np.sqrt(np.random.rand(n_points, 1)) * turns * (2 * np.pi)
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points).astype(int), np.ones(n_points).astype(int))),
    )


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
    id_list = []
    points_arr = []

    for ent in entity_points:
        if ent["ID"] in id_list:
            raise ValueError("Duplicate ID: ", ent["ID"])
        id_list.append(ent["ID"])
        points_arr.append(ent["point"])

    return id_list, np.array(points_arr)


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
    entity_labels_url: str, id_list: list, label_to_int=None, int_to_label=None
) -> (dict, List[List[float]]):
    entity_labels = list(get_label_generator(entity_labels_url))

    # Initialise label array
    labels = np.zeros(len(id_list), dtype=int)

    if label_to_int is None:
        label_to_int = dict()
    if int_to_label is None:
        int_to_label = list()

    id_to_idx = {value: idx for idx, value in enumerate(id_list)}
    for ent in entity_labels:
        label = ent["label"]
        label_str = str(label)
        if label_str not in label_to_int:
            label_to_int[label_str] = len(int_to_label)
            int_to_label.append(label)
        labels[id_to_idx[ent["ID"]]] = label_to_int[label_str]

    return labels, label_to_int, int_to_label


@CELERY.task(name=f"{QNN.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    """
    train and test a classical or dressed quantum classification network
    """

    torch.manual_seed(42)  # TODO doesn't work?
    np.random.seed(42)  # TODO doesn't work?

    # ------------------------------
    #        get input data
    # ------------------------------

    TASK_LOGGER.info(f"Starting new QNN calculation task with db id '{db_id}'")

    # load data from db
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = QNNParametersSchema().loads(task_data.parameters)

    train_points_url = input_params.train_points_url
    train_label_points_url = input_params.train_label_points_url
    test_points_url = input_params.test_points_url
    test_label_points_url = input_params.test_label_points_url
    # set variables to loaded values
    n_qubits = input_params.n_qubits  # Number of qubits
    lr = input_params.lr  # Learning rate
    batch_size = input_params.batch_size  # Numbre of samples (points) for each mini-batch
    q_depth = input_params.q_depth  # number of variational layers
    network_enum = input_params.network_enum
    # Number of optimization steps (step= 1 batch)
    epochs = input_params.epochs
    shots = input_params.shots
    optimizer = input_params.optimizer
    weight_init = input_params.weight_init
    weights_to_wiggle = input_params.weights_to_wiggle
    diff_method = input_params.diff_method
    randomly_shuffle = input_params.randomly_shuffle
    visualize = input_params.visualize
    q_device_enum = input_params.device
    ibmq_token = input_params.ibmq_token
    custom_backend = input_params.custom_backend

    TASK_LOGGER.info(f"Loaded input parameters from db: {str(input_params)}")

    if ibmq_token == "****":
        TASK_LOGGER.info(f"Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info(f"IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info(f"IBMQ_TOKEN environment variable not set")

    # load data
    train_id_list, train_data = get_indices_and_point_arr(train_points_url)
    train_labels, label_to_int, int_to_label = get_label_arr(
        train_label_points_url, train_id_list
    )

    test_id_list, test_data = get_indices_and_point_arr(test_points_url)
    test_labels = None
    if test_label_points_url != "" and test_label_points_url is not None:
        test_labels, label_to_int, int_to_label = get_label_arr(
            test_label_points_url,
            test_id_list,
            label_to_int=label_to_int,
            int_to_label=int_to_label,
        )

    # ------------------------------
    #          new qnn
    # ------------------------------
    start_time = time.time()  # Start the computation timer

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.from_numpy(train_labels)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.from_numpy(test_labels)

    # Prep data
    n_classes = len(label_to_int)
    train_dataloader = DataLoader(
        OneHotDataset(train_data, train_labels, n_classes),
        batch_size=batch_size,
        shuffle=randomly_shuffle,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preprocess_layers = [
        int(el) for el in input_params.preprocess_layers.split(",") if el != ""
    ]
    postprocess_layers = [
        int(el) for el in input_params.postprocess_layers.split(",") if el != ""
    ]
    hidden_layers = [int(el) for el in input_params.hidden_layers.split(",") if el != ""]

    model_parameters = dict(
        n_qubits=n_qubits,
        q_depth=q_depth,
        weight_init=weight_init,
        input_size=train_data.shape[1],
        output_size=n_classes,
        preprocess_layers=preprocess_layers,
        postprocess_layers=postprocess_layers,
        hidden_layers=hidden_layers,
        single_q_params=(weights_to_wiggle != 0),
        diff_method=diff_method,
    )

    if network_enum.needs_quantum_backend():
        model_parameters["quantum_device"] = q_device_enum.get_pennylane_backend(
            ibmq_token, custom_backend, n_qubits, shots
        )
    else:
        weights_to_wiggle = 0

    model = network_enum.get_neural_network(model_parameters)

    model = model.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # select optimizer
    opt = optimizer.get_optimizer(model, lr)

    # train network
    train(model, train_dataloader, loss_fn, opt, epochs, weights_to_wiggle)

    # test network
    def predictor(data: torch.Tensor) -> torch.Tensor:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        return torch.argmax(model(data), dim=1).tolist()

    predictions = [int_to_label[el] for el in predictor(test_data)]

    # Prepare labels to be saved
    output_labels = []

    for ent_id, pred in zip(test_id_list, predictions):
        output_labels.append({"ID": ent_id, "href": "", "label": pred})

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
            test_labels, predictions, int_to_label
        )

    # Output the data
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(output_labels, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "labels.json",
            "entity/label",
            "application/json",
        )

    if visualize:
        resolution = input_params.resolution
        fig = plot_data(
            train_data,
            train_id_list,
            train_labels,
            test_data,
            test_id_list,
            test_labels,
            resolution,
            predictor=predictor,
            title=plot_title,
            label_to_int=label_to_int,
        )

        # show plot
        with SpooledTemporaryFile(mode="wt") as output:
            html = fig.to_html()
            output.write(html)

            STORE.persist_task_result(
                db_id,
                output,
                "classification_plot.html",
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
                "confusion_matrix.html",
                "plot",
                "text/html",
            )

    # prepare weights output
    weights_dict = model.state_dict()
    out_weights_dict = {}
    for key in weights_dict:
        out_weights_dict[key + ".dims"] = list(
            weights_dict[key].shape
        )  # dimensions of the weights tensor

        out_weights_dict[key] = (
            weights_dict[key].flatten().tolist()
        )  # one dimensional list of weights

    qnn_outputs = []
    qnn_outputs.append(out_weights_dict)  # TODO better solution?

    # save weights in file
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(qnn_outputs, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "qnn-weights.json",
            "qnn-weights",
            "application/json",
        )

    # save quantum circuit as qasm file
    # requires pennylane-qiskit
    if network_enum.needs_quantum_backend():
        try:
            qasm_string = model_parameters["quantum_device"]._circuit.qasm()
            with SpooledTemporaryFile(mode="w") as output:
                output.write(qasm_string)
                STORE.persist_task_result(
                    db_id,
                    output,
                    "qasm-quantum-circuit.qasm",
                    "qasm-quantum-circuit",
                    "application/qasm",
                )
        except Exception as e:
            TASK_LOGGER.error("Couldn't save circuit as qasm file")
            TASK_LOGGER.error(e)

    # Print final time
    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = round(total_time - minutes * 60)

    return "Total time: " + str(minutes) + " min, " + str(seconds) + " seconds"


# TODO Quantum layer: shift for gradient determination?
# TODO weights to wiggle: number of weights in quantum circuit to update in one optimization step. 0 means all
# TODO ouput document with details for classical network parts
# TODO default enum value for optimizer not shown in gui....


# print -> TASKLOGGER
# zero initialization => completely blue background for classical network. but works well for qnn
# cleanup and comments, documentation
# really slow with aer statevector device? (with default dataset) -> resolution
