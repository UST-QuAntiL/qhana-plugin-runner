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
    Implements the circuit from A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, “The power of
    quantum neural networks,” arXiv:2011.00027 [quant-ph], Oct. 2020, Accessed: Nov. 08, 2020. [Online]. Available:
    http://arxiv.org/abs/2011.00027.

    :param q_num: number of qubits
    :return: function that constructs the circuit
    """

    def circuit(inputs: np.tensor, weights: np.tensor) -> List[np.tensor]:
        # angle encoding of the input
        for i in range(q_num):
            qml.RX(inputs[i], wires=i)

        # RY layer
        for i in range(q_num):
            qml.RY(weights[i], wires=i)

        # CNOT layer
        for i in range(1, q_num):
            for j in range(0, i):
                qml.CNOT(wires=[j, i])

        # RY layer
        for i in range(q_num):
            qml.RY(weights[i + q_num], wires=i)

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(q_num)]

    return circuit, 2 * q_num
