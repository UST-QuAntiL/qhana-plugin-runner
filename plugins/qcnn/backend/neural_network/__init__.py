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

from enum import Enum
from torch.nn import Module


class WeightInitEnum(Enum):
    standard_normal = "standard normal"
    uniform = "uniform"
    zero = "zero"


class QCNNEnum(Enum):
    qcnn1 = "QCNN1"
    qcnn2 = "QCNN2"

    def get_neural_network(self, parameters: dict) -> Module:
        if self == QCNNEnum.qcnn1:
            from .quantum_cnn import QCNN1

            return QCNN1(**parameters)
        # elif self == QCNNEnum.dressed_quantum_net:
        #     from .quantum_networks import DressedQuantumNet
        #
        #     return DressedQuantumNet(**parameters)

    def get_qubit_need(self, image) -> bool:
        if self == QCNNEnum.qcnn1:
            from .quantum_cnn import QCNN1

            return QCNN1.number_of_qubits_needed(image)
        # elif self == QCNNEnum.dressed_quantum_net:
        #     return True
