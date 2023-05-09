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

from typing import Callable

import pennylane as qml

from ... import qnn1, qnn2, qnn3


def create_qlayer(
    constructor_func: Callable, q_num: int, dev: qml.Device
) -> qml.qnn.TorchLayer:
    """
    Input of the created quantum layer should be in the range [0, pi]. The output will be in the range [-1, 1] if the
    measurement is done with pauli-z.

    :param constructor_func: Function that constructs the circuit.
    :param q_num: Number of qubits.
    :param dev: device on which the circuits will be executed
    :return: Pennylane TorchLayer.
    """
    circ_func, param_num = constructor_func(q_num)
    qnode = qml.QNode(circ_func, dev, interface="torch")

    weight_shapes = {"weights": param_num}
    qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    return qlayer


qnn_constructors = {
    "QNN1": qnn1.constructor,
    "QNN2": qnn2.constructor,
    "QNN3": qnn3.constructor,
}
