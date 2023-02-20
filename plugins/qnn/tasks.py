# hybrid quantum neural network implemented based on:
# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
# https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html

import os
from tempfile import SpooledTemporaryFile

from typing import Optional

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
)
from plugins.qnn.backend.train_and_test import train, test
from plugins.qnn.backend.visualization import plot_classification


from pennylane import numpy as np

# Timing tool
import time

# Scikit-learn tools
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from .backend.datasets import OneHotDataset, digits2position

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

    # set variables to loaded values
    n_qubits = input_params.n_qubits  # Number of qubits
    lr = input_params.lr  # Learning rate
    batch_size = input_params.batch_size  # Numbre of samples (points) for each mini-batch
    q_depth = input_params.q_depth  # number of variational layers
    use_default_dataset = input_params.use_default_dataset
    network_enum = input_params.network_enum
    # Number of optimization steps (step= 1 batch)
    epochs = input_params.epochs
    test_percentage = input_params.test_percentage
    shots = input_params.shots
    optimizer = input_params.optimizer
    weight_init = input_params.weight_init
    weights_to_wiggle = input_params.weights_to_wiggle
    q_shifts = input_params.q_shifts
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

    # load or generate dataset
    dataset = None
    if use_default_dataset:
        # spiral dataset
        dataset = twospirals(10, turns=1.52)
    else:
        # get files
        entity_points_url = input_params.entity_points_url
        clusters_url = input_params.clusters_url

        # load data from file
        entity_points = open_url(entity_points_url).json()
        clusters_entities = open_url(clusters_url).json()

        # get data
        clusters = {}
        for ent in clusters_entities:
            clusters[ent["ID"]] = ent["cluster"]
        points = []
        labels = []
        for ent in entity_points:
            points.append(ent["point"])
            labels.append(clusters[ent["ID"]])
        dataset = (points, np.array(labels))

    # ------------------------------
    #          new qnn
    # ------------------------------
    start_time = time.time()  # Start the computation timer

    # prepare data
    X, Y = dataset

    classes = list(set(list(Y)))
    n_classes = len(classes)
    n_data = len(Y)

    X = StandardScaler().fit_transform(X)

    indices = np.arange(n_data)
    if randomly_shuffle:
        # randomly shuffle data
        np.random.shuffle(indices)
    X_shuffle = X[indices]
    Y_shuffle = Y[indices]

    # determine amount of training and test data
    n_test = int(n_data * test_percentage)  # number of test data elements
    if n_test < 1:
        n_test = 1
    if n_test > n_data - 1:
        n_test = n_data - 1
    n_train = n_data - n_test  # Number of training points
    TASK_LOGGER.info(
        f"Number of data elements: n_train = '{n_train}', n_test = '{n_test}'"
    )

    # train test split
    X_train = torch.tensor(X_shuffle[:-n_test], dtype=torch.float32)
    X_test = torch.tensor(X_shuffle[-n_test:], dtype=torch.float32)

    Y_train = torch.tensor(Y_shuffle[:-n_test])
    Y_test = torch.tensor(Y_shuffle[-n_test:])

    # Prep data
    train_dataloader = DataLoader(
        OneHotDataset(X_train, Y_train, n_classes),
        batch_size=batch_size,
        shuffle=randomly_shuffle,
    )
    Y_test_one_hot = digits2position(Y_test, n_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preprocess_layers = [
        int(el) for el in input_params.preprocess_layers.split(",") if el != ""
    ]
    postprocess_layers = [
        int(el) for el in input_params.postprocess_layers.split(",") if el != ""
    ]
    hidden_layers = [int(el) for el in input_params.hidden_layers.split(",") if el != ""]

    q_shifts = [(float(el), ) for el in q_shifts.split(",") if el != ""]
    q_shifts = None if len(q_shifts) == 0 else q_shifts

    model_parameters = dict(
        n_qubits=n_qubits,
        q_depth=q_depth,
        weight_init=weight_init,
        input_size=X.shape[1],
        output_size=n_classes,
        preprocess_layers=preprocess_layers,
        postprocess_layers=postprocess_layers,
        hidden_layers=hidden_layers,
        single_q_params=(weights_to_wiggle != 0),
        q_shifts=q_shifts,
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
    accuracy_on_test_data = test(model, X_test, Y_test_one_hot, loss_fn)

    if visualize:
        resolution = input_params.resolution
        # plot results (for grid)
        figure_main = plot_classification(
            model,
            X,
            X_train,
            X_test,
            Y_train.tolist(),
            Y_test.tolist(),
            accuracy_on_test_data,
            resolution,
        )

        # plot to html
        tmpfile = BytesIO()
        figure_main.savefig(tmpfile, format="png")
        encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
        html = "<img src='data:image/png;base64,{}'>".format(encoded)

        # show plot
        with SpooledTemporaryFile(mode="wt") as output:
            output.write(html)
            STORE.persist_task_result(
                db_id,
                output,
                "plot.html",
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
        )  #  one dimensional list of weights

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
