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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    device_name = None
    if device == DeviceEnum.default:
        device_name = "default.qubit"
    else:
        device_name = "default.qubit"
    dev = qml.device(name=device_name, wires=n_qubits, shots=shots)

    # Number of pre-processing parameters (1 matrix and 1 intercept)
    n_pre = n_qubits * (n_input_nodes + 1)
    # Number of quantum node parameters (1 row of rotations per layer)
    n_quant = max_layers * n_qubits
    # Number of post-processing parameters (1 matrix and 1 intercept)
    n_post = n_classes * (n_qubits + 1)
    # Initialize a unique vector of random parameters.
    weights_flat_0 = noise_0 * np.random.randn(n_pre + n_quant + n_post)

    # ------------------------------
    #      network definition
    # ------------------------------

    @qml.qnode(dev)
    def variationalQuantumCircuit(q_weights_flat, q_in):
        """
        The quantum circuit with a variable number of quantum layers
        """
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
        """
        A quantum circuit dressed in a classical fully connected PREprocessing
        and a classical fully-connected POSTprocessing layer
        """
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

    def cost(weights_flat, points, labels):
        predictions = [dressedQNN(weights_flat, data_point=point) for point in points]
        log_like = np.sum(predictions * labels)
        return -log_like

    # ------------------------------
    #       setup for training
    # ------------------------------

    # initialized variables for training
    X, y = dataset
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=N_test, random_state=42
    )
    opt_weights = None
    train_history = []
    train_cost_history = []
    y_train_onehot = digits2position(y_train, n_classes)
    y_test_onehot = digits2position(y_test, n_classes)

    opt = None
    if optimizer == OptimizerEnum.adagrad:
        opt = qml.AdagradOptimizer(step)
    elif optimizer == OptimizerEnum.adam:
        opt = qml.AdamOptimizer(step)
    elif optimizer == OptimizerEnum.gradient_descent:
        opt = qml.GradientDescentOptimizer(stepsize=step)
    # elif optimizer == OptimizerEnum.lie_agebra:
    #    opt = qml.LieAlgebraOptimizer(stepsize=step) # needs circuit parameter (q node)
    elif optimizer == OptimizerEnum.momentum:
        opt = qml.MomentumOptimizer(stepsize=step)
    elif optimizer == OptimizerEnum.nesterov_momentum:
        opt = qml.NesterovMomentumOptimizer(stepsize=step)
    # elif optimizer == OptimizerEnum.qng:
    #    opt = qml.QNGOptimizer(stepsize=step)  # objective function must be encoded as a single qnode
    elif optimizer == OptimizerEnum.rms:
        opt = qml.RMSPropOptimizer(stepsize=step)
    # elif optimizer == OptimizerEnum.rotosolve:
    #    opt = qml.RotosolveOptimizer(stepsize=step)
    # elif optimizer == OptimizerEnum.rotoselect:
    #    opt = qml.RotoselectOptimizer(stepsize=step)
    # elif optimizer == OptimizerEnum.shot_adaptive:
    #    opt = qml.ShotAdaptiveOptimizer(stepsize=step)

    # random start weights
    opt_weights = weights_flat_0

    offset = 0
    # ------------------------------
    #           training
    # ------------------------------
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
        # Step of optimizer
        opt_weights = opt.step(
            lambda w: cost(w, train_data_batch, train_labels_batch),
            opt_weights,
        )
        # iteration results
        training_pred = np.asarray(
            [dressedQNN(opt_weights, data_point=point) for point in train_data_batch]
        )
        train_history.append(accuracy(training_pred, train_labels_batch))
        train_cost_history.append(cost_from_output(training_pred, train_labels_batch))
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
    score = accuracy(test_pred, y_test_onehot)
    TASK_LOGGER.info("Test accuracy: %4.3f" % (score))

    # ------------------------------
    #           plotting
    # ------------------------------
    # Initialize the figure that will contain the final plots.
    figure_main = plt.figure("main", figsize=(4, 4))
    plt.clf()
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
