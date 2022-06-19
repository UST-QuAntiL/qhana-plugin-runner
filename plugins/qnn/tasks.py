# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb


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

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Plot to html
import base64
from io import BytesIO

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

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

# ------------------------------
#       one-hot encoding
# ------------------------------


def digits2position(vec_of_digits, n_positions):
    """One-hot encoding of a batch of vectors."""
    return np.eye(n_positions)[vec_of_digits]


def position2digit(exp_values):
    """Inverse of digits2position()."""
    return np.argmax(exp_values)


# ------------------------------
#   layers for quantum circuit
# ------------------------------
def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOTs."""

    # first layer of CNOTS
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

    # a second shifted layer of CNOTs
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


# -----------------------------------
#   performance evaluation functions
# -----------------------------------
def cost_from_output(net_out_list, labels):
    """Cost as a function of the list of network output"""

    log_like = np.sum(net_out_list * labels)
    return -log_like


def accuracy(predictions, labels):
    """Returns fraction of correct predictions."""

    predicted_digits = np.array([position2digit(item) for item in predictions])
    label_digits = np.array([position2digit(item) for item in labels])
    return np.sum(predicted_digits == label_digits) / len(label_digits)


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
        dataset = twospirals(2000, turns=1.52)
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
    #    define initial variables
    # ------------------------------

    n_input_nodes = len(
        dataset[0][0]
    )  # dimensions of first data element (e.g. 2 for (x,y))
    classes = [0, 1]  # Class 0 = red points. class 1 = blue points.
    n_classes = len(classes)
    noise_0 = 0.001  # Initial spread of random weight vector
    max_layers = 15  # Keep 15 even if not all are used
    h = 0.2  # Plot grid step size
    start_time = time.time()  # Start the computation timer
    cm = plt.cm.RdBu  # Test point colors
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # Train point colors

    # determine amount of training and test data
    N_tot = len(dataset[1])
    N_test = int(N_tot * test_percentage)
    if N_test < 1:
        N_test = 1
    if N_test > N_tot - 1:
        N_test = N_tot - 1
    N_train = N_tot - N_test  # Number of training points
    TASK_LOGGER.info(
        f"Number of data elements: N_train = '{N_train}', N_test = '{N_test}'"
    )

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

    # ------------------------------
    #          new qnn
    # ------------------------------

    print("HI")
    X, Y = dataset  # twospirals(200, turns=1.52)

    torch.manual_seed(42)
    np.random.seed(42)

    n_qubits = 5
    step = 0.07  # learning rate
    batch_size = 10
    q_depth = 5
    n_classes = 2

    dev = qml.device("default.qubit", wires=n_qubits)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DressedQuantumNet(n_qubits, dev, q_depth)  # n_qubits, quantum_device, q_depth
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=step)

    print(len(Y))

    n_data = len(Y)

    X = StandardScaler().fit_transform(X)

    # randomly shuffle data
    indices = np.arange(n_data)
    np.random.shuffle(indices)
    X_shuffle = X[indices]
    Y_shuffle = Y[indices]

    n_test = n_data // 10

    X_train = X_shuffle[:-n_test]
    X_test = X_shuffle[-n_test:]

    Y_train = Y_shuffle[:-n_test]
    Y_test = Y_shuffle[-n_test:]

    """
    # train network
    train(model, X_train, Y_train, loss_fn, optimizer, 2000)
    #test network
    accuracy_on_test_data = test(model, X_test, Y_test, loss_fn)
    # plot results (for grid)
    plot_classification(model, X, X_train, X_test, Y_train, Y_test, accuracy_on_test_data)
    """

    # ------------------------------
    #           plotting
    # ------------------------------
    # Initialize the figure that will contain the final plots.
    figure_main = plt.figure("main", figsize=(4, 4))

    colors = cm.get_cmap("rainbow", n_classes)

    # draw training data
    figure_main.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=Y_train,
        s=100,
        lw=0,
        vmin=0,
        vmax=n_classes,
        cmap=colors,
    )
    # mark train data
    figure_main.scatter(
        X_train[:, 0], X_train[:, 1], c="b", s=50, marker="x"
    )  # , label="train data")

    # scatter test set
    figure_main.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=Y_test,
        s=100,
        lw=0,
        vmin=0,
        vmax=n_classes,
        cmap=colors,
    )

    # create legend elements
    # classes
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=i,
            markerfacecolor=colors(i),
            markersize=10,
            ls="",
        )
        for i in range(n_classes)
    ]
    # training element crosses
    legend_elements = legend_elements + [
        Line2D(
            [0],
            [0],
            marker="x",
            color="b",
            label="train data",
            markerfacecolor="g",
            markersize=8,
            ls="",
        )
    ]

    figure_main.legend(handles=legend_elements)

    figure_main.set_title(
        "Classification \naccuracy on test data={}".format(accuracy_on_test_data)
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


# using task.py from git works
# here issues with torch?
# connection error when trying to set requirement pennylane~=0.16

# TODO install pennylane version
# requirement settin (in __init__.py) and
# poetry run flask install
