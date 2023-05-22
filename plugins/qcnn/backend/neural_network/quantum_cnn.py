# Copyright 2023 QHAna plugin runner contributors.
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

# implementation based on:
# https://github.com/XanaduAI/quantum-transfer-learning/blob/master/dressed_circuit.ipynb
# https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html

from enum import Enum

import pennylane as qml

# PyTorch
import torch
import torch.nn as nn
from torch import Tensor

from . import WeightInitEnum

from typing import List, Iterator

from abc import ABCMeta, abstractmethod

from .utils import grouper


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


class QuantumCNN(nn.Module, metaclass=ABCMeta):
    def __init__(self, quantum_device: qml.Device):
        super(QuantumCNN, self).__init__()
        self.quantum_device = quantum_device

    def set_quantum_backend(self, quantum_device: qml.Device):
        self.quantum_device = quantum_device

    @abstractmethod
    def get_quantum_parameters(self):
        """Returns the quantum parameters"""

    def quanv(self, image: Tensor) -> Tensor:
        """Convolves the input image with many applications of the same quantum circuit."""
        out = torch.empty((image.shape[0] // 2, image.shape[1] // 2, image.shape[2] * 4))
        for j in range(0, image.shape[0], 2):
            for k in range(0, image.shape[1], 2):
                q_results = torch.empty((4 * image.shape[2]))
                for channel in range(image.shape[2]):
                    q_results[4 * channel + 0] = image[j, k, channel]
                    q_results[4 * channel + 1] = image[j, k + 1, channel]
                    q_results[4 * channel + 2] = image[j + 1, k, channel]
                    q_results[4 * channel + 3] = image[j + 1, k + 1, channel]
                q_results = self.circuit(q_results)()

                out[j // 2, k // 2] = q_results

        return out

    def get_representative_circuit(self, image) -> str:
        """
        Returns a qasm string of the quantum circuit used in this quantum neural network
        """
        subimage = torch.empty((4 * image.shape[2]))
        for channel in range(image.shape[2]):
            subimage[4 * channel + 0] = image[0, 0, channel]
            subimage[4 * channel + 1] = image[0, 1, channel]
            subimage[4 * channel + 2] = image[1, 0, channel]
            subimage[4 * channel + 3] = image[1, 1, channel]
        qnode = self.circuit(subimage)
        qnode.construct([], {})
        return qnode.qtape.to_openqasm()


# dressed quantum circuit
class QCNN1(QuantumCNN):
    """
    Torch module implementing the dressed quantum net.
    """

    def __init__(
        self,
        n_qubits: int,
        num_layers: int,
        quantum_device: qml.Device,
        weight_init: WeightInitEnum,
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

        # weight init
        if weight_init == WeightInitEnum.standard_normal:
            params = 0.01 * torch.randn(num_layers, n_qubits)
        elif weight_init == WeightInitEnum.uniform:
            params = 0.01 * torch.rand(num_layers, n_qubits)
        elif weight_init == WeightInitEnum.zero:
            params = torch.zeros(num_layers, n_qubits)
        else:
            raise NotImplementedError("Unknown weight init method")

        if single_q_params:
            self.params = [[nn.Parameter(el, requires_grad=True) for el in layer_params] for layer_params in params]
            for layer_idx, layer_params in enumerate(self.params):
                for i, p in enumerate(layer_params):
                    self.register_parameter(f"params({layer_idx}, {i})", p)
        else:
            self.params = nn.Parameter(params, requires_grad=True)

        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.diff_method = diff_method.get_value_for_pennylane()

    def circuit(self, data: Tensor) -> qml.QNode:
        def quantum_net():
            """
            The variational quantum circuit.
            """
            for layer_params in self.params:
                for j in range(self.n_qubits):
                    qml.RY(torch.pi * data[j], wires=j)

                for j, p in enumerate(layer_params):
                    qml.RX(p, wires=j)
                for j in range(self.n_qubits - 1):
                    qml.CNOT(wires=[j, j + 1])

            # Expectation values in the Z basis
            return [qml.expval(qml.PauliZ(wires=qubit)) for qubit in range(self.n_qubits)]

        return qml.QNode(
            quantum_net,
            self.quantum_device,
            interface="torch",
            diff_method=self.diff_method,
        )

    def forward(self, images: Tensor):
        """
        pass image through quantum cnn layer
        """
        out_tensor = torch.empty(
            (
                images.shape[0],
                images.shape[1] // 2,
                images.shape[2] // 2,
                images.shape[3] * 4,
            )
        )
        for idx, image in enumerate(images):
            out_tensor[idx] = self.quanv(image)
        return out_tensor

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def get_quantum_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())

    @staticmethod
    def number_of_qubits_needed(image: Tensor):
        return 4 * image.shape[2]
