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

import numpy as np
from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import List, Tuple


def count_wires(wires: List[List]) -> int:
    return sum(len(w) for w in wires)


class QkNN(metaclass=ABCMeta):
    def __init__(self, train_data, train_labels, k, backend):
        if not isinstance(train_data, np.ndarray):
            train_data = np.array(train_data, dtype=int)
        self.train_data = np.array(train_data)
        self.train_labels = np.array(train_labels)
        self.unique_labels = list(set(self.train_labels))

        self.k = min(k, len(self.train_data))
        self.backend = backend

    def set_quantum_backend(self, backend):
        self.backend = backend

    def get_representative_circuit(self, X) -> str:
        return ""

    @abstractmethod
    def label_point(self, X) -> int:
        """
        Takes new point x as input and returns a label for it.
        """

    def label_points(self, X) -> List[int]:
        if self.backend is None:
            raise ValueError("The quantum backend may not be None!")
        new_labels = []
        for x in X:
            new_labels.append(self.label_point(x))
        return new_labels

    @abstractmethod
    def heatmap_meaningful(self) -> bool:
        """
        Determines, if a heatmap as a background, when visualizing labeled data is meaningful or not.
        E.g. a heatmap for binary data is not meaningful.
        """

    @abstractmethod
    def get_necessary_wires(self, **kwargs) -> Tuple:
        """
        Returns the amount of necessary wires as a tuple.
        """


class QkNNEnum(Enum):
    schuld_qknn = "schuld qknn"
    simple_hamming_qknn = "simple hamming qknn"
    simple_fidelity_qknn = "simple fidelity qknn"
    simple_angle_qknn = "simple angle qknn"
    basheer_hamming_qknn = "basheer hamming qknn"

    def get_qknn_and_total_wires(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        k: int,
        max_wires: int,
        exp_itr: int = 10,
        use_access_wires: bool = True,
    ) -> Tuple[QkNN, int]:
        if self == QkNNEnum.schuld_qknn:
            from .schuld_hamming import SchuldQkNN

            wires, access_wires = self.check_and_get_qubits(
                SchuldQkNN, max_wires, train_data=train_data, train_labels=train_labels
            )
            if use_access_wires:
                wires[2] = wires[2] + access_wires

            return SchuldQkNN(
                train_data, train_labels, wires[0], wires[1], wires[2], None
            ), count_wires(wires)
        elif self == QkNNEnum.simple_hamming_qknn:
            from .simpleQkNN import SimpleHammingQkNN

            wires, access_wires = self.check_and_get_qubits(
                SimpleHammingQkNN, max_wires, train_data=train_data
            )
            if use_access_wires:
                wires[1] = wires[1] + access_wires

            return SimpleHammingQkNN(
                train_data, train_labels, k, wires[0], wires[1], None
            ), count_wires(wires)
        elif self == QkNNEnum.simple_fidelity_qknn:
            from .simpleQkNN import SimpleFidelityQkNN

            wires, access_wires = self.check_and_get_qubits(
                SimpleFidelityQkNN, max_wires, train_data=train_data
            )
            if use_access_wires:
                wires[4] = wires[4] + access_wires

            return SimpleFidelityQkNN(
                train_data,
                train_labels,
                k,
                wires[0],
                wires[1],
                wires[2],
                wires[3],
                wires[4],
                None,
            ), count_wires(wires)
        elif self == QkNNEnum.simple_angle_qknn:
            from .simpleQkNN import SimpleAngleQkNN

            wires, access_wires = self.check_and_get_qubits(
                SimpleAngleQkNN, max_wires, train_data=train_data
            )
            if use_access_wires:
                wires[4] = wires[4] + access_wires

            return SimpleAngleQkNN(
                train_data, train_labels, k, wires[0], wires[1], wires[2], wires[3], None
            ), count_wires(wires)
        elif self == QkNNEnum.basheer_hamming_qknn:
            from .basheer_hamming import BasheerHammingQkNN

            wires, access_wires = self.check_and_get_qubits(
                BasheerHammingQkNN, max_wires, train_data=train_data
            )
            if use_access_wires:
                wires[2] = wires[2] + access_wires

            return BasheerHammingQkNN(
                train_data,
                train_labels,
                k,
                wires[0],
                wires[1],
                wires[2],
                None,
                exp_itr=exp_itr,
            ), count_wires(wires)

    def check_and_get_qubits(
        self, qknn_class: QkNN, max_wires: int, **kwargs
    ) -> Tuple[List[List[int]], List[int]]:
        num_necessary_wires = qknn_class.get_necessary_wires(**kwargs)
        num_total_wires = 0
        wires = []
        for num_wires in num_necessary_wires:
            wires.append(list(range(num_total_wires, num_total_wires + num_wires)))
            num_total_wires += num_wires
        if num_total_wires > max_wires:
            raise ValueError(
                "The quantum circuit needs at least "
                + str(num_total_wires)
                + " qubits, but it only got "
                + str(max_wires)
                + "!"
            )
        access_wires = list(range(num_total_wires, max_wires))
        return wires, access_wires
