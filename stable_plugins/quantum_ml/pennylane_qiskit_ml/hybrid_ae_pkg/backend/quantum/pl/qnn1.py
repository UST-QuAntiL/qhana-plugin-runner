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

from .two_qubit_gate import add_two_qubit_gate


def constructor(q_num: int) -> Tuple[Callable, int]:
    """
    Implements circuit A from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
    of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

    :param q_num:
    :return: function that constructs the circuit
    """

    def circuit(inputs: np.tensor, weights: np.tensor) -> List[np.tensor]:
        # angle encoding of the input
        for i in range(q_num):
            qml.RX(inputs[i], wires=i)

        idx = 0  # current index for the weights

        # general two-qubit gates between every combination of qubits
        for i in range(q_num - 1):
            for j in range(q_num - 1 - i):
                add_two_qubit_gate(j, j + i + 1, weights[15 * idx : 15 * idx + 15])
                idx += 1

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(q_num)]

    param_num = 0

    for i in range(q_num - 1):
        for j in range(q_num - 1 - i):
            param_num += 15

    return circuit, param_num
