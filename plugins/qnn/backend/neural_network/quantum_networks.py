# implementation based on:
# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
# https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html

from enum import Enum

import pennylane as qml
from pennylane import numpy as np

# PyTorch
import torch
import torch.nn as nn

from plugins.qnn.schemas import WeightInitEnum

from typing import List, Iterator, Tuple

from abc import ABCMeta, abstractmethod

from .utils import grouper
from .classical_networks import create_fully_connected_net


class DiffMethodEnum(Enum):
    best = "Best"
    parameter_shift = "Parameter Shift"
    finite_diff = "Finite Differences"

    def get_value_for_pennylane(self):
        if self == DiffMethodEnum.best:
            return "best"
        elif self == DiffMethodEnum.parameter_shift:
            return "parameter-shift"
        elif self == DiffMethodEnum.finite_diff:
            return "finite-diff"


class QuantumNet(nn.Module, metaclass=ABCMeta):
    def __init__(self, quantum_device: qml.Device):
        super(QuantumNet, self).__init__()
        self.quantum_device = quantum_device

    def set_quantum_backend(self, quantum_device: qml.Device):
        self.quantum_device = quantum_device

    @abstractmethod
    def get_quantum_parameters(self):
        """Returns the quantum parameters"""

    @abstractmethod
    def get_representative_circuit(self) -> str:
        """Returns the quantum circuits used, as a qasm string"""


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
class DressedQuantumNet(QuantumNet):
    """
    Torch module implementing the dressed quantum net.
    """

    def __init__(
        self,
        input_size,
        output_size,
        n_qubits,
        quantum_device,
        q_depth,
        weight_init,
        preprocess_layers: List[int],
        postprocess_layers: List[int],
        diff_method: DiffMethodEnum,
        single_q_params: bool = False,
        **kwargs,
    ):
        """
        Initialize network with preprocessing, quantum and postprocessing layers

        n_qubits: number of qubits
        quantum_device: device for quantum network
        q_depth: amount of quantum layers
        weight_init: type of (random) initialization of the models weights (WeightInitEnum)
        """

        super().__init__(quantum_device)
        self.pre_net = create_fully_connected_net(input_size, preprocess_layers, n_qubits)
        self.add_module("output_act_func", nn.Tanh())
        self.post_net = create_fully_connected_net(
            n_qubits, postprocess_layers, output_size
        )

        # weight init
        if weight_init == WeightInitEnum.standard_normal:
            q_params = 0.01 * torch.randn(q_depth * n_qubits)
            init_fn = nn.init.normal_
        elif weight_init == WeightInitEnum.uniform:
            q_params = 0.01 * torch.rand(q_depth * n_qubits)
            init_fn = nn.init.uniform_
        elif weight_init == WeightInitEnum.zero:
            q_params = torch.zeros(q_depth * n_qubits)
            init_fn = nn.init.zeros_
        else:
            raise NotImplementedError("Unknown weight init method")

        for name, module in self.pre_net.named_modules():
            if "layer" in name:
                init_fn(module.weight)
        for name, module in self.post_net.named_modules():
            if "layer" in name:
                init_fn(module.weight)

        if single_q_params:
            self.q_params = [nn.Parameter(el, requires_grad=True) for el in q_params]
            for i in range(len(self.q_params)):
                self.register_parameter(f"q_params{i}", self.q_params[i])
        else:
            self.q_params = nn.Parameter(q_params, requires_grad=True)

        @qml.qnode(
            quantum_device,
            interface="torch",
            diff_method=diff_method.get_value_for_pennylane(),
        )
        def quantum_net(q_input_features, q_weights_flat):
            """
            The variational quantum circuit.
            """

            # Reshape weights
            q_weights = grouper(iter(q_weights_flat), n_qubits)

            # Start from state |+> , unbiased w.r.t. |0> and |1>
            H_layer(n_qubits)

            # Embed features in the quantum node
            RY_layer(q_input_features)

            # Sequence of trainable variational layers
            for k in range(q_depth):
                entangling_layer(n_qubits)
                RY_layer(next(q_weights))

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
        q_in = pre_out * torch.pi / 2.0

        # Apply the quantum circuit to each element of the batch and append to q_out
        q_out = torch.Tensor(input_features.shape[0], self.n_qubits)
        for idx, elem in enumerate(q_in):
            q_out_elem = self.q_net(elem, self.q_params).float().unsqueeze(0)
            q_out[idx] = q_out_elem

        # two-dimensional prediction from the postprocessing layer
        return self.post_net(q_out)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def get_quantum_parameters(self):
        return self.q_params

    def get_representative_circuit(self) -> str:
        x = torch.rand(self.n_qubits) * 2*torch.pi
        circuit = self.q_net
        circuit.construct([], {"q_input_features": x, "q_weights_flat": self.q_params})
        return circuit.qtape.to_openqasm()
