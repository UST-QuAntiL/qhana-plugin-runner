# implementation based on:
# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
# https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html

import pennylane as qml
from pennylane import numpy as np

# PyTorch
import torch
import torch.nn as nn

from plugins.qnn.schemas import WeightInitEnum

# define quantum layers
def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT."""
    # In other words it should apply something like :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])


# dressed quantum circuit
class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the dressed quantum net.
    """

    def __init__(self, n_qubits, quantum_device, q_depth, weight_init):
        """
        Initialize network with preprocessing, quantum and postprocessing layers

        n_qubits: number of qubits
        quantum_device: device for quantum network
        q_depth: amount of quantum layers
        weight_init: type of (random) initialization of the models weights (WeightInitEnum)
        """

        super().__init__()
        self.pre_net = nn.Linear(2, n_qubits)
        self.post_net = nn.Linear(n_qubits, 2)

        q_params = None

        # weight init
        # TODO what about biases?
        if weight_init == WeightInitEnum.standard_normal:
            q_params = 0.01 * torch.randn(q_depth * n_qubits)
            nn.init.normal_(self.pre_net.weight)
            nn.init.normal_(self.post_net.weight)
        elif weight_init == WeightInitEnum.uniform:
            q_params = 0.01 * torch.rand(q_depth * n_qubits)
            nn.init.uniform_(self.pre_net.weight)
            nn.init.uniform_(self.post_net.weight)
        elif weight_init == WeightInitEnum.zero:
            q_params = torch.zeros(q_depth * n_qubits)
            nn.init.zeros_(self.pre_net.weight)
            nn.init.zeros_(self.post_net.weight)
        else:
            print("unknown weight init method")

        self.q_params = nn.Parameter(q_params)

        # define circuit
        @qml.qnode(quantum_device, interface="torch")
        def quantum_net(q_input_features, q_weights_flat):
            """
            The variational quantum circuit.
            """

            # Reshape weights
            q_weights = q_weights_flat.reshape(q_depth, n_qubits)

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            H_layer(n_qubits)

            # Embed features in the quantum node
            RY_layer(q_input_features)

            # Sequence of trainable variational layers
            for k in range(q_depth):
                entangling_layer(n_qubits)
                RY_layer(q_weights[k])

            # Expectation values in the Z basis
            exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
            return tuple(exp_vals)

        self.q_net = quantum_net
        self.n_qubits = n_qubits

    def forward(self, input_features):
        """
        pass input features through preprocessing, quantum and postprocessing layers
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        # q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = (
                self.q_net(elem, self.q_params).float().unsqueeze(0)
            )  # quantum_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))

        # two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)


class ClassicalNet(nn.Module):
    """
    Torch module implementing the classical net.
    """

    def __init__(self, n_features, depth, weight_init):
        """
        Initialize network with preprocessing, classical and postprocessing layers

        n_features: number of features per layer
        depth: number of layers
        weight_init: type of (random) initialization of the models weights (WeightInitEnum)
        """

        super().__init__()

        self.pre_net = nn.Linear(2, n_features)
        self.classical_net = nn.ModuleList()
        for i in range(depth):
            self.classical_net.append(nn.Linear(n_features, n_features))
            self.classical_net.append(nn.ReLU())  # TODO change?
        # self.classical_net = nn.ModuleList(
        #     [nn.Linear(n_features, n_features) for i in range(depth)]
        #     # TODO (n_features * n_features) features instead of n_features features?
        # )  # TODO how many layers?

        self.post_net = nn.Linear(n_features, 2)

        # weight initialization
        self.weight_init = weight_init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # init weights according to initialization type
            if self.weight_init == WeightInitEnum.standard_normal:
                module.weight.data.normal_(mean=0.0, std=1.0)
            elif self.weight_init == WeightInitEnum.uniform:
                module.weight.data.uniform_()
            elif self.weight_init == WeightInitEnum.zero:  # TODO plot is completely blue?
                module.weight.data.zero_()
            else:
                print("unknown weight init method")

            # initialize bias
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_features):
        """
        pass input features through classical layers
        """

        # preprocessing layer
        pre_out = self.pre_net(input_features)
        out = torch.tanh(pre_out)

        # classical net
        for i, layer in enumerate(self.classical_net):
            out = layer(out)
        c_out = torch.tanh(out)

        # postprocessing layer
        return self.post_net(c_out)
