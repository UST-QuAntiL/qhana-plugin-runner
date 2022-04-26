# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb


from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)

from plugins.qnn import QNN
from plugins.qnn.schemas import InputParameters, QNNParametersSchema
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask

from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.requests import open_url

from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)


# Plotting
#%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# PennyLane
import pennylane as qml
from pennylane import numpy as np

# Optimized logsumexp().
# from scipy.misc import logsumexp      # Working but deprecated
# from scipy.special import logsumexp   # May gives problems with autograd.

# Adam optimizer
from pennylane.optimize import AdamOptimizer

# Timing tool
import time

# Scikit-learn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def digits2position(vec_of_digits, n_positions):
    """One-hot encoding of a batch of vectors."""
    return np.eye(n_positions)[vec_of_digits]


def position2digit(exp_values):
    """Inverse of digits2position()."""
    return np.argmax(exp_values)


# def H_layer(nqubits):
#    """Layer of single-qubit Hadamard gates."""
#    for idx in range(nqubits):
#        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOTs."""
    # In other words it should apply something like :
    # CNOT CNOT CNOT CNOT... CNOT
    #  CNOT CNOT CNOT... CNOT

    # Loop over even indices: i=0,2,...N-2
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

    # Loop over odd indices: i=1,3,...N-3
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def cost_from_output(weights_flat, net_out_list, labels):
    """Cost as a function of the list of network output"""

    log_like = np.sum(net_out_list * labels)
    return -log_like


def accuracy(predictions, labels):
    """Returns fraction of correct predictions."""

    predicted_digits = np.array([position2digit(item) for item in predictions])
    label_digits = np.array([position2digit(item) for item in labels])
    return np.sum(predicted_digits == label_digits) / len(label_digits)


@CELERY.task(name=f"{QNN.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new QNN calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)
    input_params: InputParameters = QNNParametersSchema().loads(task_data.parameters)
    entity_points_url = input_params.entity_points_url
    TASK_LOGGER.info(
        f"Loaded input parameters from db: entity_points_url='{entity_points_url}'"
    )
    clusters_url = input_params.clusters_url
    TASK_LOGGER.info(f"Loaded input parameters from db: clusters_url='{clusters_url}'")
    # load data from file
    entity_points = open_url(entity_points_url).json()
    clusters_entities = open_url(clusters_url).json()

    n_qubits = input_params.n_qubits  # Number of qubits
    step = input_params.step  # Learning rate
    batch_size = input_params.batch_size  # Numbre of samples (points) for each mini-batch
    q_depth = (
        input_params.q_depth
    )  # Depth of the quantum circuit (number of variational layers)
    n_input_nodes = 2  # 2 input nodes (x and y coordinates of data points).
    classes = [0, 1]  # Class 0 = red points. class 1 = blue points.
    n_classes = len(classes)
    N_total_iterations = (
        input_params.N_total_iterations  # 1000                 # Number of optimization steps (step= 1 batch)
    )
    noise_0 = 0.001  # Initial spread of random weight vector
    max_layers = 15  # Keep 15 even if not all are used
    h = 0.2  # Plot grid step size
    start_time = time.time()  # Start the computation timer
    cm = plt.cm.RdBu  # Test point colors
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # Train point colors

    # get data
    clusters = {}
    for ent in clusters_entities:
        clusters[ent["ID"]] = ent["cluster"]
    points = []
    labels = []
    for ent in entity_points:
        points.append(ent["point"])
        labels.append(clusters[ent["ID"]])
    TASK_LOGGER.info(f"Points '{len(points)}'")
    TASK_LOGGER.info(f"Labels '{len(labels)}'")

    # determine amount of training and test data
    N_test = int(len(labels) * input_params.test_percentage)
    TASK_LOGGER.info(f"N_test before '{N_test}'")
    if N_test < 1:
        N_test = 1
    if N_test > len(labels) - 1:
        N_test = len(labels) - 1
    TASK_LOGGER.info(f"N_test after '{N_test}'")

    N_train = len(labels) - N_test  # Number of training points
    TASK_LOGGER.info(f"N_train '{N_train}'")

    dev = qml.device("default.qubit", wires=n_qubits)

    # Number of pre-processing parameters (1 matrix and 1 intercept)
    n_pre = n_qubits * (n_input_nodes + 1)
    # Number of quantum node parameters (1 row of rotations per layer)
    n_quant = max_layers * n_qubits
    # Number of post-processing parameters (1 matrix and 1 intercept)
    n_post = n_classes * (n_qubits + 1)
    # Initialize a unique vector of random parameters.
    weights_flat_0 = noise_0 * np.random.randn(n_pre + n_quant + n_post)

    @qml.qnode(dev)
    def variationalQuantumCircuit(q_weights_flat, q_in):
        q_weights = q_weights_flat.reshape(max_layers, n_qubits)
        # Hadamard layer
        for idx in range(n_qubits):
            qml.Hadamard(wires=idx)
        # Embedding layer
        RY_layer(q_in)
        # (multiple) trainable variational layers
        for k in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[k + 1])
        # Measure in Z basis
        exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
        return tuple(exp_vals)

    def dressedQNN(weights_flat, data_point=None):
        # ------------------------------
        # classical preprocessing layer
        # ------------------------------
        # prepare weights
        pre_weights_flat = weights_flat[:n_pre]  # preprocessing weights
        pre_weights = pre_weights_flat.reshape(n_qubits, n_input_nodes + 1)
        # affine operation and non-linear activation
        pre_out = np.tanh(np.dot(pre_weights[:, :-1], data_point) + pre_weights[:, -1])
        # ------------------------------
        #        quantum layers
        # ------------------------------
        # prepare weights
        q_weights_flat = weights_flat[n_pre : n_pre + n_quant]  # quantum weights
        q_in = pre_out * np.pi / 2.0
        q_out = variationalQuantumCircuit(q_weights_flat, q_in)
        # ------------------------------
        # classical postprocessing layer
        # ------------------------------
        # prepare weights
        post_weights_flat = weights_flat[n_pre + n_quant :]  # postprocessing weights
        post_weights = post_weights_flat.reshape(n_classes, n_qubits + 1)
        # affine operation
        post_one = np.dot(post_weights[:, :-1], q_out) + post_weights[:, -1]
        post_out = post_one - np.log(np.sum(np.exp(post_one), axis=0))
        return post_out

    def cost(weights_flat, points, labels, node=None):
        predictions = [dressedQNN(weights_flat, data_point=point) for point in points]
        log_like = np.sum(predictions * labels)
        return -log_like

    datasets = (points, np.array(labels))

    X, y = datasets
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_test, random_state=42
    )
    opt_weights = None
    train_history = []
    train_cost_history = []
    y_train_onehot = digits2position(y_train, n_classes)
    y_test_onehot = digits2position(y_test, n_classes)
    opt = qml.GradientDescentOptimizer(stepsize=step)
    # random start weights
    opt_weights = weights_flat_0

    offset = 0
    for it in range(N_total_iterations):
        start_it_time = time.time()
        # reshuffle if all training data was used
        if offset > N_train - 1:
            indices = np.arange(N_train)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            y_train_onehot = y_train_onehot[indices]
            offset = 0
        # select training data batch
        train_data_batch = X_train[offset : offset + batch_size]
        train_labels_batch = y_train_onehot[offset : offset + batch_size]
        # Step of Adam optimizer
        opt_weights = opt.step(
            lambda w: cost(w, train_data_batch, train_labels_batch),
            opt_weights,
        )
        # opt_weights = opt.step(lambda w: cost_function(w, train_data_batch, train_labels_batch, node="quantum"), opt_weights)
        # iteration results
        training_pred = np.asarray(
            [dressedQNN(opt_weights, data_point=point) for point in train_data_batch]
        )
        # training_pred = np.asarray(
        #    [full_network(opt_weights, data_point=point, node="quantum") for point in train_data_batch]
        # )
        train_history.append(accuracy(training_pred, train_labels_batch))
        train_cost_history.append(
            cost_from_output(opt_weights, training_pred, train_labels_batch)
        )
        # print iteration info
        total_it_time = time.time() - start_it_time
        minutes_it = total_it_time // 60
        seconds_it = round(total_it_time - minutes_it * 60)
        TASK_LOGGER.info(
            "Iteration: %4d of %4d. Time:%3d min %3d sec."
            % (it + 1, N_total_iterations, minutes_it, seconds_it),
        )
        offset += batch_size
    # ------------------------------
    #         test network
    # ------------------------------
    test_pred = np.asarray(
        [dressedQNN(opt_weights, data_point=point) for point in X_test]
    )
    # test_pred = np.asarray(
    #    [full_network(opt_weights, data_point=point, node="quantum") for point in X_test]
    # )
    score = accuracy(test_pred, y_test_onehot)
    TASK_LOGGER.info("Test accuracy: %4.3f" % (score))

    # ------------------------------
    #           plotting
    # ------------------------------
    # Initialize the figure that will contain the final plots.
    figure_main = plt.figure("main", figsize=(4, 4))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # compute prediction for each gridpoint
    grid_results = np.asarray(
        [dressedQNN(opt_weights, data_point=point) for point in grid_points]
    )
    # grid_results = np.asarray( [full_network(opt_weights, data_point=point, node="quantum") for point in grid_points])
    # decision function: negative for class 0,  positive for class 1
    Z = np.tanh(grid_results[:, 1] - grid_results[:, 0])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    # Plot training points
    plt.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", alpha=0.1
    )
    # Plot test points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k")
    plt.text(
        xx.max() - 0.3,
        yy.min() + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )

    import base64
    from io import BytesIO

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
            "qnn.json",
            "qnn-weights",
            "application/json",
        )

    # Print final results
    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = round(total_time - minutes * 60)

    return "Total time: " + str(minutes) + " min, " + str(seconds) + " seconds"
