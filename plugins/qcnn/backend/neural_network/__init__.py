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
    qcnn3 = "QCNN3"
    qcnn4 = "QCNN4"
    qcnn5 = "QCNN5"
    qcnn6 = "QCNN6"
    qcnn7 = "QCNN7"

    def get_neural_network(self, parameters: dict) -> Module:
        if self == QCNNEnum.qcnn1:
            from .quantum_cnn import QCNN1

            return QCNN1(**parameters)
        elif self == QCNNEnum.qcnn2:
            from .quantum_cnn import QCNN2

            return QCNN2(**parameters)
        elif self == QCNNEnum.qcnn3:
            from .quantum_cnn import QCNN3

            return QCNN3(**parameters)
        elif self == QCNNEnum.qcnn4:
            from .quantum_cnn import QCNN4

            return QCNN4(**parameters)
        elif self == QCNNEnum.qcnn5:
            from .quantum_cnn import QCNN5

            return QCNN5(**parameters)
        elif self == QCNNEnum.qcnn6:
            from .quantum_cnn import QCNN6

            return QCNN6(**parameters)
        elif self == QCNNEnum.qcnn7:
            from .quantum_cnn import QCNN7

            return QCNN7(**parameters)

    def get_qubit_need(self, image) -> int:
        if self == QCNNEnum.qcnn1:
            from .quantum_cnn import QCNN1

            return QCNN1.number_of_qubits_needed(image)
        elif self == QCNNEnum.qcnn2:
            from .quantum_cnn import QCNN2

            return QCNN2.number_of_qubits_needed(image)
        elif self == QCNNEnum.qcnn3:
            from .quantum_cnn import QCNN3

            return QCNN3.number_of_qubits_needed(image)
        elif self == QCNNEnum.qcnn4:
            from .quantum_cnn import QCNN4

            return QCNN4.number_of_qubits_needed(image)
        elif self == QCNNEnum.qcnn5:
            from .quantum_cnn import QCNN5

            return QCNN5.number_of_qubits_needed(image)
        elif self == QCNNEnum.qcnn6:
            from .quantum_cnn import QCNN6

            return QCNN6.number_of_qubits_needed(image)
        elif self == QCNNEnum.qcnn7:
            from .quantum_cnn import QCNN7

            return QCNN7.number_of_qubits_needed(image)
