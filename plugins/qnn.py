# Copyright 2021 QHAna plugin runner contributors.
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

from enum import Enum
from http import HTTPStatus
from importlib.metadata import metadata
from sre_constants import NOT_LITERAL_IGNORE
from tempfile import SpooledTemporaryFile
from tokenize import PlainToken
from types import new_class
from typing import Mapping, Optional

import marshmallow as ma
from celery.canvas import chain
from celery.utils.log import get_task_logger
from flask import Response, redirect
from flask.app import Flask
from flask.globals import request
from flask.helpers import url_for
from flask.templating import render_template
from flask.views import MethodView
from marshmallow import EXCLUDE, post_load

from qhana_plugin_runner.requests import open_url
from qhana_plugin_runner.plugin_utils.entity_marshalling import (
    save_entities,
)
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.plugin_schemas import (
    PluginMetadataSchema,
    PluginMetadata,
    PluginType,
    EntryPoint,
    DataMetadata,
)
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    SecurityBlueprint,
)
from qhana_plugin_runner.celery import CELERY
from qhana_plugin_runner.db.models.tasks import ProcessingTask
from qhana_plugin_runner.storage import STORE
from qhana_plugin_runner.tasks import save_task_error, save_task_result
from qhana_plugin_runner.util.plugins import QHAnaPluginBase, plugin_identifier

####################################
import numpy as np

# import matplotlib.pyplot as plt

# ModuleNotFoundError: No module named 'torch'
import torch
from torch.autograd import Function

from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *


_plugin_name = "qnn"
__version__ = "v0.1.0"
_identifier = plugin_identifier(_plugin_name, __version__)


QNN_BLP = SecurityBlueprint(
    _identifier,  # blueprint name
    __name__,  # module import name!
    description="QNN plugin API",
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


##### ????????????????
class InputParameters:
    def __init__(self, theta: float, learning_rate: float, epochs: int):
        self.theta = theta
        self.learning_rate = learning_rate
        self.epochs = epochs


class QNNParametersSchema(FrontendFormBaseSchema):
    """n_qubits = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Qubits",
            "description": "Number of Qubits used for the quantum circuit",
            "input_type": "text",
        },
    )"""

    theta = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Theta (Not used right now)",
            "description": "The input parameter for the QNN (rotation parameter)",
            "input_type": "text",
        },
    )
    # step
    learning_rate = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Learning Rate",
            "description": "Learning rate for the training of the hybrid NN",
            "input_type": "text",
        },
    )
    epochs = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Epochs",
            "description": "Number of epochs for the training of the hybrid NN",
            "input_type": "text",
        },
    )
    # ?????????????????
    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


@QNN_BLP.route("/")
class PluginsView(MethodView):
    """Plugins collection resource."""

    @QNN_BLP.response(HTTPStatus.OK, PluginMetadataSchema)
    @QNN_BLP.require_jwt("jwt", optional=True)
    def get(self):
        """QNN endpoint returning the plugin metadata."""

        return PluginMetadata(
            title="Quantum Neutral Network (QNN)",
            description="Simple (hardcoded) QNN",
            name=QNN.instance.identifier,
            version=QNN.instance.version,
            type=PluginType.simple,
            entry_point=EntryPoint(
                href=url_for(f"{QNN_BLP.name}.ProcessView"),
                ui_href=url_for(f"{QNN_BLP.name}.MicroFrontend"),
                data_input=[],  # TODO ?
                data_output=[
                    """DataMetadata(
                        data_type="txt",
                        content_type=["text/plain"],
                        required=True,
                    )"""
                ],
            ),
            tags=[],  # TODO?
        )


@QNN_BLP.route("/ui/")
class MicroFrontend(MethodView):
    """Micro frontend for the QNN plugin."""

    example_inputs = {  # TODO?
        "inputStr": "Sample input string.",
    }

    @QNN_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the QNN plugin.")
    @QNN_BLP.arguments(
        QNNParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="query",
        required=False,
    )
    @QNN_BLP.require_jwt("jwt", optional=True)
    def get(self, errors):
        """Return the micro frontend."""
        return self.render(request.args, errors)

    @QNN_BLP.html_response(HTTPStatus.OK, description="Micro frontend of the QNN plugin.")
    @QNN_BLP.arguments(
        QNNParametersSchema(
            partial=True, unknown=EXCLUDE, validate_errors_as_result=True
        ),
        location="form",
        required=False,
    )
    @QNN_BLP.require_jwt("jwt", optional=True)
    def post(self, errors):
        """Return the micro frontend with prerendered inputs."""
        return self.render(request.form, errors)

    def render(self, data: Mapping, errors: dict):
        schema = QNNParametersSchema()
        return Response(
            render_template(
                "simple_template.html",
                name=QNN.instance.name,
                version=QNN.instance.version,
                schema=schema,
                values=data,
                errors=errors,
                process=url_for(f"{QNN_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{QNN_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )
        """
        data_dict = dict(data)
        fields = QNNParametersSchema().fields

        # define default values
        default_values = {
            fields["theta"].data_key: 0.0,
            fields["solver"].data_key: SolverEnum.auto,
            fields["scale"].data_key: False,
        }

        # overwrite default values with other values if possible
        default_values.update(data_dict)
        data_dict = default_values

        return Response(
            render_template(
                "simple_template.html",
                name=QNN.instance.name,
                version=QNN.instance.version,
                schema=QNNParametersSchema(),
                values=data_dict,
                errors=errors,
                process=url_for(f"{QNN_BLP.name}.ProcessView"),
                example_values=url_for(
                    f"{QNN_BLP.name}.MicroFrontend", **self.example_inputs
                ),
            )
        )
        """


@QNN_BLP.route("/process/")
class ProcessView(MethodView):
    """Start a long running processing task."""

    @QNN_BLP.arguments(QNNParametersSchema(unknown=EXCLUDE), location="form")
    @QNN_BLP.response(HTTPStatus.OK, TaskResponseSchema())
    @QNN_BLP.require_jwt("jwt", optional=True)
    def post(self, arguments):
        """Start the calculation task."""
        db_task = ProcessingTask(
            task_name=calculation_task.name,
            parameters=QNNParametersSchema().dumps(arguments),
        )
        db_task.save(commit=True)

        # all tasks need to know about db id to load the db entry
        task: chain = calculation_task.s(db_id=db_task.id) | save_task_result.s(
            db_id=db_task.id
        )
        # save errors to db
        task.link_error(save_task_error.s(db_id=db_task.id))
        task.apply_async()

        db_task.save(commit=True)

        return redirect(
            url_for("tasks-api.TaskView", task_id=str(db_task.id)), HTTPStatus.SEE_OTHER
        )


class QNN(QHAnaPluginBase):
    name = _plugin_name
    version = __version__

    def __init__(self, app: Optional[Flask]) -> None:
        super().__init__(app)

    def get_api_blueprint(self):
        return QNN_BLP

    def get_requirements(self) -> str:
        # return "torch~=1.10\nnumpy~=1.22\nqiskit~=0.34"  # TODO? # after specifying here "poetry run flask install"
        return "torch~=1.10\ntorchvision~=0.11\nnumpy~=1.22\nqiskit~=0.34\nmatplotlib~=3.5\npennylane~=0.20\nscikit-learn~=0.24.2"  # TODO? # after specifying here "poetry run flask install"


TASK_LOGGER = get_task_logger(__name__)


@CELERY.task(name=f"{QNN.instance.identifier}.calculation_task", bind=True)
def calculation_task(self, db_id: int) -> str:
    TASK_LOGGER.info(f"Starting new QNN calculation task with db id '{db_id}'")
    task_data: Optional[ProcessingTask] = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        msg = f"Could not load task data with id {db_id} to read parameters!"
        TASK_LOGGER.error(msg)
        raise KeyError(msg)

    input_params: InputParameters = QNNParametersSchema().loads(task_data.parameters)
    theta = input_params.theta
    TASK_LOGGER.info(f"Loaded input parameters from db: theta='{theta}'")
    if theta is None:
        raise ValueError("No input argument provided!")
    """if theta:
        out_str = theta
        with SpooledTemporaryFile(mode="w") as output:
            output.write(theta)
            STORE.persist_task_result(
                db_id, output, "output.txt", "qnn-output", "text/plain"
            )
        return "result: " + repr(out_str)
    return "Empty input, no output could be generated!"
    """
    # return test_quantum_circuit_class(theta)
    # return train(input_params.learning_rate, input_params.epochs)
    return run_dressed_quantum_circuit()
    # return "Task is done"


##########################################################
# qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html

# create a quantum class with qiskit
#   here: a 1-qubit circuit with one trainable quantum parameter theta
#   for simplicity: hardcoded circuit, RY-rotation by theta
#             _____     _____________        _________
#   |0>-------| H |-----| RY(theta) |--------|measure|
#             _____     _____________        _________

#   measure in z-basis and calculate the oz expectation
class QuantumCircuit:
    """
    # This class provides a simple interface for interaction with the quantum circuit
    """

    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter("theta")  # input?

        self._circuit.h(all_qubits)  # initialize qubits with Hadamard gates
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)  # encode inputdata with RY rotation

        self._circuit.measure_all()  # measure all qubits
        # ---------------------------

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit, self.backend)
        qobj = assemble(
            t_qc,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        # run
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)

        return np.array([expectation])


# test implementation
def test_quantum_circuit_class(theta):
    simulator = qiskit.Aer.get_backend("aer_simulator")

    circuit = QuantumCircuit(1, simulator, 100)
    # print("Expected value for rotation pi {}".format(circuit.run([np.pi])[0]))
    # circuit._circuit.draw()

    # TODO actually use theta
    return "Expected value for rotation %s (with given theta %s)" % (
        (circuit.run([theta])[0]),
        theta,
    )


############################################################

# quantum classical class with pytorch


# backpropagation using PyTorch
class HybridFunction(Function):
    """Hybrid quantum - classical function definition"""

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """Forward pass computation"""
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass computation"""
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])

            gradient = torch.tensor([expectation_right]) - torch.tensor(
                [expectation_left]
            )
            gradients.append(gradient)
        gradients = np.array([gradients]).T
        return torch.tensor([gradients]).float() * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Hybrid quantum - classical layer definition"""

    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


#################################
# 3.4 Data Loading and Preprocessing


def load_training_data():
    # Concentrating on the first 100 samples
    n_samples = 100

    X_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    # Leaving only labels 0 and 1
    idx = np.append(
        np.where(X_train.targets == 0)[0][:n_samples],
        np.where(X_train.targets == 1)[0][:n_samples],
    )

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)
    return train_loader
    #####
    # display data (skipped here)
    ######


def load_testing_data():
    n_samples = 50

    X_test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    idx = np.append(
        np.where(X_test.targets == 0)[0][:n_samples],
        np.where(X_test.targets == 1)[0][:n_samples],
    )

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)
    return test_loader


# 3.5 Creating the Hybrid Neural Network
# CNN with 2 fully connected layers at the end
# The value of the last neuron of the fully-connected layer is fed as parameter theta into the quantum circuit
# the circuit measurement then serves as a final prediction for 0 or 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.hybrid = Hybrid(qiskit.Aer.get_backend("aer_simulator"), 100, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), -1)


# 3.6 Training the Network
def train(learning_rate, epochs):
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 0.001)
    loss_func = nn.NLLLoss()

    # epochs = 20
    # epochs = 1  # TODO
    loss_list = []

    train_loader = load_training_data()

    model.train()
    for epoch in range(epochs):
        total_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate loss
            loss = loss_func(output, target)
            # Backward pass
            loss.backward()
            # Optimize the weights
            optimizer.step()

            total_loss.append(loss.item())
        loss_list.append(sum(total_loss) / len(total_loss))
        # print('Training [{:.0f}%]\tLoss: {:4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
        # return "Training [{:.0f}%]\tLoss: {:4f}".format(
        #    100.0 * (epoch + 1) / epochs, loss_list[-1]
        # )

        # 3.7 Testing the Network

        test_loader = load_testing_data()

        model.eval()
        with torch.no_grad():

            correct = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss = loss_func(output, target)
                total_loss.append(loss.item())

            # print(
            #     "Performance on test data: \n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
            #         sum(total_loss) / len(total_loss), correct / len(test_loader) * 100
            #     )
            # )
            return (
                "Performance on test data: \n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%".format(
                    sum(total_loss) / len(total_loss), correct / len(test_loader) * 100
                )
            )


#########################################################################################################

############################################
#       dressed quantum circuit
############################################

# Plotting
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


n_qubits = 5  # Number of qubits
step = 0.07  # Learning rate
batch_size = 10  # Numbre of samples (points) for each mini-batch
data_noise = 0.2  # Spread of the data points around the spirals
q_depth = 5  # Depth of the quantum circuit (number of variational layers)
rng_w_state = 55  # Seed for random initial weights
n_input_nodes = 2  # 2 input nodes (x and y coordinates of data points).
classes = [0, 1]  # Class 0 = red points. class 1 = blue points.
n_classes = len(classes)
N_train = 2000  # Number of training points
N_batches = N_train // batch_size  # Number of training batches
N_total_iterations = 2  # 1000  # Number of optimization steps (step= 1 batch)
noise_0 = 0.001  # Initial spread of random weight vector
N_test = 200  # Number of test points
N_tot = N_train + N_test  # Total number of points
max_layers = 15  # Keep 15 even if not all are used
h = 0.2  # Plot grid step size
start_time = time.time()  # Start the computation timer
cm = plt.cm.RdBu  # Test point colors
cm_bright = ListedColormap(["#FF0000", "#0000FF"])  # Train point colors


# initialize a PennyLane with the default simulator
dev = qml.device("default.qubit", wires=n_qubits)

# initialize vector with random weights
# Number of pre-processing parameters (1 matrix and 1 intercept)
n_pre = n_qubits * (n_input_nodes + 1)

# Number of quantum node parameters (1 row of rotations per layer)
n_quant = max_layers * n_qubits

# Number of classical node parameters (1 matrix and 1 intercept)
n_class = n_qubits * (n_qubits + 1)

# Number of post-processing parameters (1 matrix and 1 intercept)
n_post = n_classes * (n_qubits + 1)

# Set seed of random number generator
# rng_w = np.random.RandomState(rng_w_state)
rng_w = np.random.default_rng(rng_w_state)

# Initialize a unique vector of random parameters.
# weights_flat_0 = noise_0 * rng_w.randn(n_pre + n_quant + n_class + n_post)
weights_flat_0 = noise_0 * rng_w.random(n_pre + n_quant + n_class + n_post)

classifiers = ["Classical", "Quantum"]
names = ["Entirely classical", "Dressed quantum"]


##############################
# Synthetic benchmark dataset
##############################


def twospirals(n_points, noise=0.7, turns=1.52, random_state=None):
    """Returns the two spirals dataset."""

    if random_state == None:
        rng_sp = np.random
    else:
        # rng_sp = np.random.RandomState(random_state)
        rng_sp = np.random.default_rng(random_state)
    # n = np.sqrt(rng_sp.rand(n_points, 1)) * turns * (2 * np.pi)
    n = np.sqrt(rng_sp.random((n_points, 1))) * turns * (2 * np.pi)
    # d1x = -np.cos(n) * n + rng_sp.rand(n_points, 1) * noise
    d1x = -np.cos(n) * n + rng_sp.random((n_points, 1)) * noise
    # d1y = np.sin(n) * n + rng_sp.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + rng_sp.random((n_points, 1)) * noise
    return (
        np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
        np.hstack((np.zeros(n_points).astype(int), np.ones(n_points).astype(int))),
    )


datasets = [
    twospirals(N_tot, random_state=21, turns=1.52),
    twospirals(N_tot, random_state=21, turns=2.0),
]


def digits2position(vec_of_digits, n_positions):
    """One-hot encoding of a batch of vectors."""
    return np.eye(n_positions)[vec_of_digits]


def position2digit(exp_values):
    """Inverse of digits2position()."""
    return np.argmax(exp_values)


##############################
# take a look at the datasets
##############################

# TODO use
def plot_datasets():
    figure_dataset = plt.figure("dataset", figsize=(4, 4 * len(datasets)))
    for ds_cnt, ds in enumerate(datasets):

        # Normalize dataset and split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=N_test, random_state=42
        )

        # Plot the dataset
        ax = plt.subplot(len(datasets), 1, ds_cnt + 1)
        ax.set_title("Dataset %d" % (ds_cnt + 1))

        # Plot training points
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.1,
        )

        # Plot test points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k")


#################################
# dressed quantum circuit
#################################


def pre_processing_net(pre_weights_flat, data_point=None):
    """Classical layer which preprocesses data. The output
    should be a feature vector suitable to be injected into a quantum network.
    """
    # Reshape weights
    pre_weights = pre_weights_flat.reshape(n_qubits, n_input_nodes + 1)

    # Affine operation
    pre_one = np.dot(pre_weights[:, :-1], data_point) + pre_weights[:, -1]

    # Non-linear activation
    pre_out = np.tanh(pre_one)

    return pre_out


def post_processing_net(post_weights_flat, post_in):
    """Classical layer which postprocesses data. The input
    should be a vector of expectation values of a quantum network.
    """
    # Reshape weights
    post_weights = post_weights_flat.reshape(n_classes, n_qubits + 1)

    # Affine operation
    post_one = np.dot(post_weights[:, :-1], post_in) + post_weights[:, -1]

    # LogSumExp normalization layer. Choose between scipy or NumPy functions.

    # With scipy logsumexp
    # post_out = post_one - logsumexp(post_one, axis=0, keepdims=True)

    # Directly with NumPy functions
    post_out = post_one - np.log(np.sum(np.exp(post_one), axis=0))

    return post_out


# quantum layers


def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


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


@qml.qnode(dev)
def q_net(q_weights_flat, q_in):

    # Reshape weights
    q_weights = q_weights_flat.reshape(max_layers, n_qubits)

    # Start from unbiased |+> state w.r.t. |0> and |1>
    H_layer(n_qubits)

    # Embed features in the quantum node
    RY_layer(q_in)

    # Sequence of trainable variational layers
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k + 1])

    # Expectation values in the Z basis.
    # exp_vals = [qml.expval.PauliZ(position) for position in range(n_qubits)]
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


# classical
def c_net(c_weights_flat, c_in):
    """Classical network which should replace the quantum node in order to make
    a relatively fair classical/quantum comparison.
    """

    # Reshape weights
    c_weights = c_weights_flat.reshape(n_qubits, n_qubits + 1)

    # Classical layers
    c_one = np.dot(c_weights[:, :-1], c_in) + c_weights[:, -1]
    return np.tanh(c_one)


# network
def full_network(weights_flat, data_point=None, node="quantum"):
    """Full neural network including all quantum and classical layers."""

    # Split weight vector into four vectors:
    # Pre-processing layer parameters
    pre_weights_flat = weights_flat[:n_pre]

    # Quantum circuit parameters
    q_weights_flat = weights_flat[n_pre : n_pre + n_quant]

    # Classical benchmark parameters
    c_weights_flat = weights_flat[n_pre + n_quant : n_pre + n_quant + n_class]

    # Post-processing layer parameters
    post_weights_flat = weights_flat[n_pre + n_quant + n_class :]

    # Application of classical pre-processing layer
    pre_out = pre_processing_net(pre_weights_flat, data_point=data_point)

    # Quantum circuit
    if node == "quantum":
        # Rescale [-1,1] to [-pi/2,pi/2]
        q_in = pre_out * np.pi / 2.0
        net_out = q_net(q_weights_flat, q_in)

    # Classical benchmark
    if node == "classical":
        net_out = c_net(c_weights_flat, pre_out)

    # Application of classical post-processing layer
    post_out = post_processing_net(post_weights_flat, net_out)
    return post_out


# cost and accuracy
def cost_function(weights_flat, points, labels, node=None):
    """Objective function to be minimized by the training process"""

    predictions = [
        full_network(weights_flat, data_point=point, node=node) for point in points
    ]
    log_like = np.sum(predictions * labels)
    return -log_like


def cost_from_output(weights_flat, net_out_list, labels):
    """Cost as a function of the list of network output"""

    log_like = np.sum(net_out_list * labels)
    return -log_like


def accuracy(predictions, labels):
    """Returns fraction of correct predictions."""

    predicted_digits = np.array([position2digit(item) for item in predictions])
    label_digits = np.array([position2digit(item) for item in labels])
    return np.sum(predicted_digits == label_digits) / len(label_digits)


#########################
# RUN
#########################
def run_dressed_quantum_circuit():
    # training and results
    # Initialize the figure that will contain the final plots.
    figure_main = plt.figure(
        "main", figsize=(4 * (len(classifiers) + 1), 4 * len(datasets))
    )

    i = 1
    # Iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # Normalize dataset and split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=N_test, random_state=42
        )

        # Plot settings
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # First, just plot the dataset
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")

        # Plot training points
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.1,
        )

        # Plot test points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # Iterate over classifiers
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        for name, clf in zip(names, classifiers):
            print("Training ", name, "...")
            node = None
            if clf == "Quantum":
                node = "quantum"
            if clf == "Classical":
                node = "classical"

            # ===============================
            #       Netrwork training
            # ===============================

            opt_weights = None
            # rng_shuff = np.random.RandomState(150)
            rng_shuff = np.random.default_rng(150)
            train_history = []
            train_cost_history = []
            y_train_onehot = digits2position(y_train, n_classes)
            y_test_onehot = digits2position(y_test, n_classes)
            opt = AdamOptimizer(step)

            # Start with random weigths
            opt_weights = weights_flat_0

            # Training iteration loop
            offset = 0
            for it in range(N_total_iterations):
                start_it_time = time.time()

                # If all train data has been used, then reshuffle
                if offset > N_train - 1:
                    indices = np.arange(N_train)
                    rng_shuff.shuffle(indices)
                    X_train = X_train[indices]
                    y_train = y_train[indices]
                    y_train_onehot = y_train_onehot[indices]
                    offset = 0

                train_data_batch = X_train[offset : offset + batch_size]
                train_labels_batch = y_train_onehot[offset : offset + batch_size]

                # Step of Adam optimizer
                opt_weights = opt.step(
                    lambda w: cost_function(
                        w, train_data_batch, train_labels_batch, node=node
                    ),
                    opt_weights,
                )

                # Iteration results
                training_pred = np.asarray(
                    [
                        full_network(opt_weights, data_point=point, node=node)
                        for point in train_data_batch
                    ]
                )
                train_history.append(accuracy(training_pred, train_labels_batch))
                train_cost_history.append(
                    cost_from_output(opt_weights, training_pred, train_labels_batch)
                )

                # Print info for each iteration
                total_it_time = time.time() - start_it_time
                minutes_it = total_it_time // 60
                seconds_it = round(total_it_time - minutes_it * 60)
                print(
                    "Iteration: %4d of %4d. Time:%3d min %3d sec."
                    % (it + 1, N_total_iterations, minutes_it, seconds_it),
                    end="\r",
                    flush=True,
                )
                offset += batch_size

            # ================================
            #        Netrwork testing
            # ================================

            print("\nComputing accuracy on test data...")
            test_pred = np.asarray(
                [
                    full_network(opt_weights, data_point=point, node=node)
                    for point in X_test
                ]
            )
            score = accuracy(test_pred, y_test_onehot)
            print("Test accuracy: %4.3f " % (score))

            # ======================
            #        Plotting
            # ======================

            # In order to draw the decision border we
            # compute predictions for the whole 2D plane.
            grid_results = np.asarray(
                [
                    full_network(opt_weights, data_point=point, node=node)
                    for point in grid_points
                ]
            )

            # Decision function: negative for class 0, positive for class 1.
            Z = np.tanh(grid_results[:, 1] - grid_results[:, 0])
            Z = Z.reshape(xx.shape)

            # Add subplot to the main figure.
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

            # Plot training points
            ax.scatter(
                X_train[:, 0],
                X_train[:, 1],
                c=y_train,
                cmap=cm_bright,
                edgecolors="k",
                alpha=0.1,
            )

            # Plot test points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k"
            )
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

            # Add model names as titles
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                xx.max() - 0.3,
                yy.min() + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1

    # Print final results
    total_time = time.time() - start_time
    minutes = total_time // 60
    seconds = round(total_time - minutes * 60)
    print("Total time: ", minutes, " min, ", seconds, " seconds")
    print("\n")

    # Show the final figure
    plt.tight_layout()
    plt.show()
    print("\n")

    return "YES!"


# ----------------------------------
# dressed quantum circuit proposed in https://arxiv.org/pdf/1912.08278.pdf
# ----------------------------------
# classify points (in spirals) in 2 classes
# input 2 real coordinates, 2 real variables as output (one-hot encoding the blue and red classes)
# model: dressed quanrum circuit: L4->2 ° Q ° L2->4
#   L2->4: classical layer with Ln0-n1: x->y=tanh(Wx+b)
#   Q: (bare) variational quantum circuit
#       Q = M ° Q_ ° E_
#           E_: x -> |x> = E(x)|0>      EMBEDDING LAYER
#               HERE: prepares each qubit in a balanced superposition of |0> and |1> and then rotation around y axis parameterized by x
#           Q_ = Lq ° ... ° L2 ° L1     TRAINABLE CIRCUIT
#               HERE: 5 variational layers Q_ = L5°L4°L3°L2°L1
#               where L(w):|x>->|y> = K X Ry(wk)|x>     (??)
#               and K is an entangling unitary operation made of 3 CNOT gates: K=(CNOT X I_3,4)(I_1,2 X CNOT)(I_1 X CNOT X I_4)
#                       X means tensorproduct?
#           M: |x> -> y = <x|y^|x>      MEASUREMENT LAYER
#               HERE: expectation value of the Z = dias(1, -1) Pauli matrix, locally estimated for each qubit:
#                               |  <y| Z X I X I X I | y>  |
#                               |  <y| I X Z X I X I | y>  |
#                 M(|y>) = y =  |  <y| I X I X Z X I | y>  |
#                               |  <y| I X I X I X Z | y>  |
#   L4->2: linear classical layer without activation: Ln0-n1: x->y=Wx+b
# ---------------------------
#   Given an input point of coordinates x = (x1, x2), the classification is done according to argmax(y) where y = (y1, y2) is the output of the dressed quantum circuit
#   Loss function: cross entropy (implicitly preceded by a LogSoftMax layer)  J = -E Sum(j){ pj * log(pj_^)}        where pj is the true probability distribution and pj_^ is the predicted one. Average E
#       minimized via the Adam optimizer
#   LogSoftMax layer: yi -> pj_^ = e^( (yj) / (e^y1 + e^y2) ) such that the predicted distribution {pj_^} is a valid probability distribution for j=1,2
# ---------------------------
# entirely classical counterpart replaces quantum circuit by a classical layer: C = L4->2 ° L4->4 ° L2->4


# specify parameters
#   dataelements to classify
#


# quantum neural network
#   Initialize (balanced superposition)
#   Encode data (rotation)
#   Successively apply sequence of trainable rotation layers and constant entangling layers
#   measure local expectation value of the Z operator of each qubit -> classical output vector


# old qhana
#   QNN.py : classes
#       NNQuantumCircuit
#           - __init__
#           - run
#       HybridFunction
#           - forward
#           - backward
#       Hybrid
#           - __init__
#           - update_epoch
#           - init_weights
#           - forward
#       DressedQNN
#           - __init__
#           - update_epoch
#           - forward
#   QNNCircuitGenerator.py
#       QNNCircuitGenerator
#           - genCircuit
#   QNNcircuitExecutor.py
#       CircuitExecutor
#           - runCircuit
#           - parametrization_from_parameter_names


# readme (old qhana)
# classification : once clusters have been created, we can perform some classification tasks. The following clustering algorithms are implemented in QHAna
#       ClassicalSklearnNN      Classical neural network method using SKLearn [22]
#       HybridQNN               Hybrid quantum-classical Neutal Network using PyTorch and Qiskit [23]


# TODO linter
# flake8
# einstellungen > settings: Python linting flake8 enabled []
# if enabled: errors in qnn.py?
# because imports should all be at the top and all variables should be used
