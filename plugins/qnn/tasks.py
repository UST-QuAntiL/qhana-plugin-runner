# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb


import imp
from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)

from plugins.qnn import QNN
from plugins.qnn.schemas import (
    DeviceEnum,
    InputParameters,
    OptimizerEnum,
    QNNParametersSchema,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url

from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)
from plugins.qnn.backend.model import DressedQuantumNet
from plugins.qnn.backend.train_and_test import train, test
from plugins.qnn.backend.visualization import plot_classification


# PennyLane
import pennylane as qml
from pennylane import numpy as np

# Timing tool
import time

# Scikit-learn tools
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# Plotting
import matplotlib.pyplot as plt

# Plot to html
import base64
from io import BytesIO

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
    TASK_LOGGER.info(f"Loaded input parameters from db: n_qubits='{n_qubits}'")
    step = input_params.step  # Learning rate
    TASK_LOGGER.info(f"Loaded input parameters from db: step='{step}'")
    batch_size = input_params.batch_size  # Numbre of samples (points) for each mini-batch
    TASK_LOGGER.info(f"Loaded input parameters from db: batch_size='{batch_size}'")
    q_depth = (
        input_params.q_depth
    )  # Depth of the quantum circuit (number of variational layers)
    TASK_LOGGER.info(f"Loaded input parameters from db: q_depth='{q_depth}'")
    use_default_dataset = input_params.use_default_dataset
    TASK_LOGGER.info(
        f"Loaded input parameters from db: use_default_dataset='{use_default_dataset}'"
    )
    N_total_iterations = (
        input_params.N_total_iterations  # Number of optimization steps (step= 1 batch)
    )
    TASK_LOGGER.info(
        f"Loaded input parameters from db: N_total_iterations='{N_total_iterations}'"
    )
    test_percentage = input_params.test_percentage
    TASK_LOGGER.info(
        f"Loaded input parameters from db: test_percentage='{test_percentage}'"
    )
    shots = input_params.shots
    TASK_LOGGER.info(f"Loaded input parameters from db: shots='{shots}'")
    optimizer = input_params.optimizer
    TASK_LOGGER.info(f"Loaded input parameters from db: optimizer='{optimizer}'")
    device = input_params.device
    TASK_LOGGER.info(f"Loaded input parameters from db: device='{device}'")

    # load or generate dataset
    dataset = None
    if use_default_dataset:
        print("Use default dataset")
        # spiral dataset
        dataset = twospirals(200, turns=1.52)  # twospirals(2000, turns=1.52)
    else:
        print("Load dataset from files")
        # get data from file
        entity_points_url = input_params.entity_points_url
        TASK_LOGGER.info(
            f"Loaded input parameters from db: entity_points_url='{entity_points_url}'"
        )
        clusters_url = input_params.clusters_url
        TASK_LOGGER.info(
            f"Loaded input parameters from db: clusters_url='{clusters_url}'"
        )
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

    X, Y = dataset

    classes = list(set(list(Y)))
    n_classes = len(classes)

    torch.manual_seed(42)
    np.random.seed(42)

    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dressed quantum network
    model = DressedQuantumNet(n_qubits, dev, q_depth)  # n_qubits, quantum_device, q_depth
    model = model.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer TODO choose based on input parameter
    optimizer = optim.Adam(model.parameters(), lr=step)

    n_data = len(Y)
    print(n_data)

    X = StandardScaler().fit_transform(X)

    # randomly shuffle data
    indices = np.arange(n_data)
    np.random.shuffle(indices)
    X_shuffle = X[indices]
    Y_shuffle = Y[indices]

    # n_test = n_data // 10

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
    X_train = X_shuffle[:-n_test]
    X_test = X_shuffle[-n_test:]

    Y_train = Y_shuffle[:-n_test]
    Y_test = Y_shuffle[-n_test:]

    print("Y_train len", len(Y_train))
    print("Y_test len", len(Y_test))

    # train network
    train(
        model,
        X_train,
        Y_train,
        loss_fn,
        optimizer,
        N_total_iterations,
        n_classes,
        batch_size,
    )
    # test network
    accuracy_on_test_data = test(model, X_test, Y_test, loss_fn, n_classes)
    # plot results (for grid)
    figure_main = plot_classification(
        model, X, X_train, X_test, Y_train, Y_test, accuracy_on_test_data
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

    # save weights
    # prepare output
    # -----------------------------------------
    #              save weights
    # -----------------------------------------
    n_input_nodes = len(
        dataset[0][0]
    )  # dimensions of first data element (e.g. 2 for (x,y))
    noise_0 = 0.001  # Initial spread of random weight vector
    max_layers = 15  # Keep 15 even if not all are used
    # Number of pre-processing parameters (1 matrix and 1 intercept)
    n_pre = n_qubits * (n_input_nodes + 1)
    # Number of quantum node parameters (1 row of rotations per layer)
    n_quant = max_layers * n_qubits
    # Number of post-processing parameters (1 matrix and 1 intercept)
    n_post = n_classes * (n_qubits + 1)
    # Initialize a unique vector of random parameters.
    weights_flat_0 = noise_0 * np.random.randn(n_pre + n_quant + n_post)

    # random start weights
    opt_weights = weights_flat_0
    # in json format
    # nd array cant be saved in json format
    qnn_outputs = []
    qnn_outputs.append(
        {
            # weights of preprocessing layer
            "pre_weight_flat": opt_weights[:n_pre].tolist(),
            "pre_weights_dims": (n_qubits, n_input_nodes + 1),
            # weights of quantum layer
            "q_weights_flat": opt_weights[n_pre : n_pre + n_quant].tolist(),
            "q_weights_dims": (max_layers, n_qubits),
            # weight of postprocessing layer
            "post_weights_flat": opt_weights[n_pre + n_quant :].tolist(),
            "post_weights_dims": (n_classes, n_qubits + 1),
        }
    )

    # save weights
    with SpooledTemporaryFile(mode="w") as output:
        save_entities(qnn_outputs, output, "application/json")
        STORE.persist_task_result(
            db_id,
            output,
            "qnn-weights.json",
            "qnn-weights",
            "application/json",
        )

    # Print final results
    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = round(total_time - minutes * 60)

    # TASK_LOGGER.info("Quantum circuit:", qml.draw(variationalQuantumCircuit)) # not possible with this qml version?
    return "Total time: " + str(minutes) + " min, " + str(seconds) + " seconds"


# TODO quantum devices
# TODO optimizers
# TODO save actual weights to file
# TODO visualize circuit??
# TODO cleanup and comments
# TODO documentation
# TODO add references
# TODO turn off gradient calculation for some parts?
# TODO option to initialize wrights (different random inits, zero init, with weights file)
