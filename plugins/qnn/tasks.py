# hybrid quantum neural network implemented based on:
# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
# https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html

import os
from tempfile import SpooledTemporaryFile

from typing import Optional

from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)

from plugins.qnn import QNN
from plugins.qnn.schemas import (
    QuantumBackends,
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


def get_optimizer(optimizer, model, step):
    if optimizer == OptimizerEnum.adadelta:
        TASK_LOGGER.info("adadelta")
        return optim.Adadelta(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.adagrad:
        TASK_LOGGER.info("adagrad")
        return optim.Adagrad(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.adam:
        TASK_LOGGER.info("adam")
        return optim.Adam(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.adamW:
        TASK_LOGGER.info("adamW")
        return optim.AdamW(model.parameters(), lr=step)
    elif (
        optimizer == OptimizerEnum.sparse_adam
    ):  # TODO check "RuntimeError: SparseAdam does not support dense gradients, please consider Adam instead"
        TASK_LOGGER.info("SparseAdam")
        return optim.SparseAdam(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.adamax:
        TASK_LOGGER.info("Adamax")
        return optim.Adamax(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.asgd:
        TASK_LOGGER.info("ASGD")
        return optim.ASGD(model.parameters(), lr=step)
    elif (
        optimizer == OptimizerEnum.lbfgs
    ):  # TODO step() missing 1 required argument: 'closure'
        TASK_LOGGER.info("LBFGS")
        return optim.LBFGS(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.n_adam:
        TASK_LOGGER.info("NAdam")
        return optim.NAdam(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.r_adam:
        TASK_LOGGER.info("RAdam")
        return optim.RAdam(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.rms_prob:
        TASK_LOGGER.info("RMSprop")
        return optim.RMSprop(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.Rprop:
        TASK_LOGGER.info("Rprop")
        return optim.Rprop(model.parameters(), lr=step)
    elif optimizer == OptimizerEnum.sdg:
        TASK_LOGGER.info("SGD")
        return optim.SGD(model.parameters(), lr=step)
    else:
        TASK_LOGGER.error("unknown optimizer")
        return -1


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
    ibmq_token = input_params.ibmq_token
    TASK_LOGGER.info(f"Loaded input parameters from db: ibmq_token")

    if ibmq_token == "****":
        TASK_LOGGER.info(f"Loading IBMQ token from environment variable")

        if "IBMQ_TOKEN" in os.environ:
            ibmq_token = os.environ["IBMQ_TOKEN"]
            TASK_LOGGER.info(f"IBMQ token successfully loaded from environment variable")
        else:
            TASK_LOGGER.info(f"IBMQ_TOKEN environment variable not set")

    custom_backend = input_params.custom_backend
    TASK_LOGGER.info(
        f"Loaded input parameters from db: custom_backend='{custom_backend}'"
    )

    weight_init = input_params.weight_init
    TASK_LOGGER.info(f"Loaded input parameters from db: weight_init='{weight_init}'")
    randomly_shuffle = input_params.randomly_shuffle
    TASK_LOGGER.info(
        f"Loaded input parameters from db: randomly_shuffle='{randomly_shuffle}'"
    )

    # load or generate dataset
    dataset = None
    if use_default_dataset:
        TASK_LOGGER.info("Use default dataset")
        # spiral dataset
        dataset = twospirals(200, turns=1.52)  # TODO set number of data elements in GUI??
    else:
        TASK_LOGGER.info("Load dataset from files")
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

    torch.manual_seed(42)  # TODO doesn't work?
    np.random.seed(42)  # TODO doesn't work?

    # choose quantum backend
    dev = QuantumBackends.get_pennylane_backend(
        device, ibmq_token, custom_backend, n_qubits, shots
    )
    #  dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dressed quantum network
    model = DressedQuantumNet(n_qubits, dev, q_depth, weight_init)
    model = model.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # select optimizer
    opt = get_optimizer(optimizer, model, step)

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
    X_train = X_shuffle[:-n_test]
    X_test = X_shuffle[-n_test:]

    Y_train = Y_shuffle[:-n_test]
    Y_test = Y_shuffle[-n_test:]

    # train network
    train(
        model,
        X_train,
        Y_train,
        loss_fn,
        opt,
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

    # model.state_dict()
    #   q_params
    #   pre_net.weight
    #   pre_net.bias
    #   post_net.weight
    #   post_net.bias

    # prepare output
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

    # TODO Qasm output (requires pygments>2.4 library)

    # save quantum circuit as qasm file
    # requires pennylane-qiskit
    # TODO check if it works (try catch????)
    try:
        with SpooledTemporaryFile(mode="wt") as output:
            qasm_str = (
                dev._circuit.qasm()
            )  # formatted=True) # formatted only works with pygments>2.4
            output.write(qasm_str)

            STORE.persist_task_result(
                db_id,
                output,
                "qasm-quantum-circuit.html",
                "qasm-quantum-circuit",
                "text/html",
            )
    except Exception as e:
        print("Couldn't save circuit as qasm file")  # TODO
        print(e)  # TODO

    # Print final time
    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = round(total_time - minutes * 60)

    return "Total time: " + str(minutes) + " min, " + str(seconds) + " seconds"


# TODO cleanup and comments, documentation
# TODO hidden layers for preprocessing and postprocessing? GUI..
# TODO Quantum layer: shift for gradient determination?
# TODO weights to wiggle: number of weights in quantum circuit to update in one optimization step. 0 means all
# TODO check if quantum device selection works as intended
# TODO check if training still works
# TODO ouput document with details for classical network parts
# TODO error: cannot import name Basebackend from qiskit.providers

# DONE save actual weights to file
# DONE add references
# DONE optimizers
#       old qhana: Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, SGD, Rprop, RMSprop, LBFGS
#       default: Adam
# DONE option to initialize weights (different random inits, zero init, with weights file)
#   standard_normal, uniform, zero (distribution for (random) initialization of weights)
#   default: uniform
# DONE quantum devices
#   QuantumBackend
#       enum default: aer_statevector_simulator (aer_qasm_simulator)
#   IBMQ-Custom-Backend: str default "", name of a custom backend of ibmq
#   IBMQ-Token: str default "", IBMQ-Token for access to IBMQ online service
# DONE prints -> task logger info
# DONE visualize circuit?? openqasm file?
# DONE turn off gradient calculation for some parts?