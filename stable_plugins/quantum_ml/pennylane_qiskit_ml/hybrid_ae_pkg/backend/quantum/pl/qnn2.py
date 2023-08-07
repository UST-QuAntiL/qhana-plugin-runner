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

from typing import Tuple, Callable, List

import pennylane as qml
import pennylane.numpy as np


def constructor(q_num: int) -> Tuple[Callable, int]:
    """
    Implements circuit B from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
    of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

    :param q_num: number of qubits
    :return:
    """

    def circuit(inputs: np.tensor, weights: np.tensor) -> List[np.tensor]:
        # angle encoding of the input
        for i in range(q_num):
            qml.RX(inputs[i], wires=i)

        idx = 0  # current index for the weights

        # layer of single qubit rotations
        for i in range(q_num):
            qml.Rot(weights[idx], weights[idx + 1], weights[idx + 2], wires=i)
            idx += 3

        # layer of controlled single qubit rotations
        for i in range(q_num):
            for j in range(q_num):
                if i != j:
                    qml.CRot(
                        weights[idx], weights[idx + 1], weights[idx + 2], wires=[i, j]
                    )
                    idx += 3

        # layer of single qubit rotations
        for i in range(q_num):
            qml.Rot(weights[idx], weights[idx + 1], weights[idx + 2], wires=i)
            idx += 3

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(q_num)]

    return circuit, (2 * q_num + q_num * (q_num - 1)) * 3
