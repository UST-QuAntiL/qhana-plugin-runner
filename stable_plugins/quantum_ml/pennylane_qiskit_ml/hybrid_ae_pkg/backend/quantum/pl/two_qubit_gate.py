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

import pennylane as qml
import pennylane.numpy as np


def add_two_qubit_gate(q1, q2, weights: np.tensor):
    """
    Adds a general two-qubit gate.
    Implements a general two-qubit gate as seen in F. Vatan and C. Williams, “Optimal Quantum Circuits for General
    Two-Qubit Gates,” Phys. Rev. A, vol. 69, no. 3, p. 032315, Mar. 2004, doi: 10.1103/PhysRevA.69.032315.

    :param q1: first input qubit for the gate
    :param q2: second input qubit for the gate
    :param weights: parameters of the gate
    """

    qml.U3(weights[0], weights[1], weights[2], wires=q1)
    qml.U3(weights[3], weights[4], weights[5], wires=q2)

    qml.CNOT(wires=[q2, q1])

    qml.RZ(weights[6], wires=q1)
    qml.RY(weights[7], wires=q2)

    qml.CNOT(wires=[q1, q2])
    qml.RY(weights[8], wires=q2)
    qml.CNOT(wires=[q2, q1])

    qml.U3(weights[9], weights[10], weights[11], wires=q1)
    qml.U3(weights[12], weights[13], weights[14], wires=q2)
