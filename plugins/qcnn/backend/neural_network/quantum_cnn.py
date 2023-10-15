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

from abc import ABCMeta


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

    def quanv(self, image: Tensor) -> Tensor:
        """Convolves the input image with many applications of the same quantum circuit."""
        out = torch.empty(
            (image.shape[0] // 2, image.shape[1] // 2, image.shape[2] * 4),
            device=image.device,
        )
        for j in range(0, image.shape[0], 2):
            for k in range(0, image.shape[1], 2):
                q_results = torch.empty((4 * image.shape[2]))
                for channel in range(image.shape[2]):
                    q_results[4 * channel + 0] = image[j, k, channel]
                    q_results[4 * channel + 1] = image[j, k + 1, channel]
                    q_results[4 * channel + 2] = image[j + 1, k, channel]
                    q_results[4 * channel + 3] = image[j + 1, k + 1, channel]
                q_results = torch.tensor(self.circuit(q_results)(), device=image.device)

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

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def get_quantum_parameters(self) -> List[nn.Parameter]:
        return list(self.parameters())


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
            self.params = [
                [nn.Parameter(el, requires_grad=True) for el in layer_params]
                for layer_params in params
            ]
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
            ),
            device=images.device,
        )
        for idx, image in enumerate(images):
            out_tensor[idx] = self.quanv(image)
        return out_tensor

    @staticmethod
    def number_of_qubits_needed(image: Tensor) -> int:
        return 4 * image.shape[2]


class QCNN2(QuantumCNN):
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
            params = 0.01 * torch.randn(num_layers, n_qubits, 3)
        elif weight_init == WeightInitEnum.uniform:
            params = 0.01 * torch.rand(num_layers, n_qubits, 3)
        elif weight_init == WeightInitEnum.zero:
            params = torch.zeros(num_layers, n_qubits, 3)
        else:
            raise NotImplementedError("Unknown weight init method")

        if single_q_params:
            self.params = [
                [
                    [nn.Parameter(el, requires_grad=True) for el in u_params]
                    for u_params in layer_params
                ]
                for layer_params in params
            ]
            for layer_idx, layer_params in enumerate(self.params):
                for i, u_params in enumerate(layer_params):
                    for axis, p in enumerate(u_params):
                        self.register_parameter(f"params({layer_idx}, {i}, {axis})", p)
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
                for j, d in enumerate(data[: self.n_qubits]):
                    qml.RX(torch.pi * d, wires=j)

                for j, p in enumerate(layer_params):
                    qml.Rot(p[0], p[1], p[2], wires=j)
                    for j in range(self.n_qubits):
                        if j == 0:
                            for i in range(self.n_qubits - 1):
                                qml.CZ(wires=[j, i + 1])
                        elif j == 1:
                            qml.CZ(wires=[j, j - 1])
                            for i in range(self.n_qubits - 2):
                                qml.CZ(wires=[j, i + 2])

                        elif j == 2:
                            qml.CZ(wires=[j, j + 1])
                            for i in range(self.n_qubits - 1, 1, -1):
                                qml.CZ(wires=[j, i - 2])
                        elif j == 3:
                            for i in range(self.n_qubits - 1, 0, -1):
                                qml.CZ(wires=[j, i - 1])

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

    @staticmethod
    def number_of_qubits_needed(image: Tensor) -> int:
        return 4 * image.shape[2]


class QCNN3(QuantumCNN):
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
            params = 0.01 * torch.randn(num_layers, 2, n_qubits)
        elif weight_init == WeightInitEnum.uniform:
            params = 0.01 * torch.rand(num_layers, 2, n_qubits)
        elif weight_init == WeightInitEnum.zero:
            params = torch.zeros(num_layers, 2, n_qubits)
        else:
            raise NotImplementedError("Unknown weight init method")

        if single_q_params:
            self.params = [
                [
                    [nn.Parameter(el, requires_grad=True) for el in q_params]
                    for q_params in layer_params
                ]
                for layer_params in params
            ]
            for layer_idx, layer_params in enumerate(self.params):
                for i, q_params in enumerate(layer_params):
                    for j, p in enumerate(q_params):
                        self.register_parameter(f"params({layer_idx}, {i}, {j})", p)
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
                for i, d in enumerate(data[: self.n_qubits]):
                    qml.RX(torch.pi * data[i], wires=i)

                for i, p in enumerate(layer_params[0]):
                    qml.RX(p, wires=i)
                    qml.RZ(p, wires=i)
                for i, p in enumerate(layer_params[1]):
                    qml.CRX(
                        p,
                        wires=[
                            self.n_qubits - 1 - i,
                            (self.n_qubits - 2 - i) % self.n_qubits,
                        ],
                    )

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

    @staticmethod
    def number_of_qubits_needed(image: Tensor) -> int:
        return 4 * image.shape[2]


class QCNN4(QuantumCNN):
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
            self.params = [
                [nn.Parameter(el, requires_grad=True) for el in layer_params]
                for layer_params in params
            ]
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
                for i, d in enumerate(data[: self.n_qubits]):
                    qml.RY(torch.pi * d, wires=i)

                for _ in range(2):
                    for i, p in enumerate(layer_params):
                        qml.RX(p, wires=i)
                        qml.RZ(p, wires=i)

                    for i in range(0, self.n_qubits - 1, 2):
                        qml.CRX(layer_params[i], wires=[i, i + 1])

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

    @staticmethod
    def number_of_qubits_needed(image: Tensor) -> int:
        return 4 * image.shape[2]


class QCNN5(QuantumCNN):
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
            params = 0.01 * torch.randn(num_layers, n_qubits, 3)
        elif weight_init == WeightInitEnum.uniform:
            params = 0.01 * torch.rand(num_layers, n_qubits, 3)
        elif weight_init == WeightInitEnum.zero:
            params = torch.zeros(num_layers, n_qubits, 3)
        else:
            raise NotImplementedError("Unknown weight init method")

        if single_q_params:
            self.params = [
                [
                    [nn.Parameter(el, requires_grad=True) for el in u_params]
                    for u_params in layer_params
                ]
                for layer_params in params
            ]
            for layer_idx, layer_params in enumerate(self.params):
                for i, u_params in enumerate(layer_params):
                    for j, p in enumerate(u_params):
                        self.register_parameter(f"params({layer_idx}, {i}, {j})", p)
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
                for i, d in enumerate(data[: self.n_qubits]):
                    qml.RY(torch.pi * d, wires=i)

                for j, p in enumerate(layer_params):
                    qml.U3(p[0], p[1], p[2], wires=j)
                for i in range(0, self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for j in range(0, self.n_qubits - 1, 2):
                    qml.RY(layer_params[j][0], wires=j)
                    qml.RZ(layer_params[j][0], wires=j + 1)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for j in range(0, self.n_qubits, 2):
                    qml.RY(layer_params[j][0], wires=j)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for j, p in enumerate(layer_params):
                    qml.U3(p[0], p[1], p[2], wires=j)

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

    @staticmethod
    def number_of_qubits_needed(image: Tensor) -> int:
        return 4 * image.shape[2]


class QCNN6(QuantumCNN):
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
            self.params = [
                [nn.Parameter(el, requires_grad=True) for el in layer_params]
                for layer_params in params
            ]
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
                for i, d in enumerate(data[: self.n_qubits]):
                    qml.RY(torch.pi * d, wires=i)

                for idx in range(self.n_qubits):
                    qml.Hadamard(wires=idx)
                for i in range(self.n_qubits - 1, 0, -1):
                    qml.CZ(wires=[i, i - 1])
                for i, p in enumerate(layer_params):
                    qml.RX(p, wires=i)

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

    @staticmethod
    def number_of_qubits_needed(image: Tensor) -> int:
        return 4 * image.shape[2]


class QCNN7(QuantumCNN):
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
            params = 0.01 * torch.randn(num_layers, n_qubits, 3)
        elif weight_init == WeightInitEnum.uniform:
            params = 0.01 * torch.rand(num_layers, n_qubits, 3)
        elif weight_init == WeightInitEnum.zero:
            params = torch.zeros(num_layers, n_qubits, 3)
        else:
            raise NotImplementedError("Unknown weight init method")

        if single_q_params:
            self.params = [
                [
                    [nn.Parameter(el, requires_grad=True) for el in u_params]
                    for u_params in layer_params
                ]
                for layer_params in params
            ]
            for layer_idx, layer_params in enumerate(self.params):
                for i, u_params in enumerate(layer_params):
                    for j, p in enumerate(u_params):
                        self.register_parameter(f"params({layer_idx}, {i}, {j})", p)
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
                for i, d in enumerate(data[: self.n_qubits]):
                    qml.RY(torch.pi * d, wires=i)

                for i, p in enumerate(layer_params):
                    qml.U1(p[0], wires=i)

                for i in range(self.n_qubits - 1, -1, -1):
                    qml.CNOT(wires=[(i + 1) % self.n_qubits, i])

                for i, p in enumerate(layer_params):
                    qml.U3(p[1], p[0], p[2], wires=i)

                for i in range(self.n_qubits - 1, -1, -1):
                    qml.CNOT(wires=[(i + 1) % self.n_qubits, i])

                for i, p in enumerate(layer_params):
                    qml.U1(p[1], wires=i)

                for i in range(self.n_qubits - 1, -1, -1):
                    qml.CNOT(wires=[(i + 1) % self.n_qubits, i])

                for i, p in enumerate(layer_params):
                    qml.U3(p[0], p[1], p[2], wires=i)

                for i in range(self.n_qubits - 1, -1, -1):
                    qml.CNOT(wires=[(i + 1) % self.n_qubits, i])

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

    @staticmethod
    def number_of_qubits_needed(image: Tensor) -> int:
        return 4 * image.shape[2]
