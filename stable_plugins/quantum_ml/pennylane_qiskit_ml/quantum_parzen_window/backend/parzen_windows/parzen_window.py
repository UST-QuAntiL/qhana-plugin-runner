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
from pennylane import Device


def count_wires(wires: List[List]) -> int:
    return sum(len(w) for w in wires)


class ParzenWindow(metaclass=ABCMeta):
    def __init__(
        self, train_data, train_labels, distance_threshold: float, backend: Device
    ):
        self.backend = backend
        if not isinstance(train_data, np.ndarray):
            train_data = np.array(train_data, dtype=int)
        self.train_data = train_data
        self.train_labels = train_labels
        self.unique_labels = list(set(self.train_labels))
        self.distance_threshold = distance_threshold

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
        new_labels = []
        for x in X:
            new_labels.append(self.label_point(x))
        return new_labels

    @abstractmethod
    def get_necessary_wires(self, **kwargs) -> Tuple:
        """
        Returns the amount of necessary wires as a tuple.
        """


class QParzenWindowEnum(Enum):
    ruan_window = "ruan parzen window"

    def get_parzen_window(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        window_size: int,
        max_wires: int,
        use_access_wires: bool = True,
    ) -> Tuple[ParzenWindow, int]:
        if self == QParzenWindowEnum.ruan_window:
            from .ruan_parzen_window import RuanParzenWindow

            wires, access_wires = self.check_and_get_qubits(
                RuanParzenWindow,
                max_wires,
                train_data=train_data,
                train_labels=train_labels,
            )
            if use_access_wires:
                wires[2] = wires[2] + access_wires

            return RuanParzenWindow(
                train_data, train_labels, window_size, wires[0], wires[1], wires[2], None
            ), count_wires(wires)

    def check_and_get_qubits(
        self, parzen_window_class: ParzenWindow, max_wires: int, **kwargs
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Given a quantum parzen window instance and the maximum number of qubits, this method returns the necessary
        qubits needed and a list of the unnecessary qubits. This function throws an error, if the amount of necessary
        qubits is greater than the maximum number of qubits.
        """
        num_necessary_wires = parzen_window_class.get_necessary_wires(**kwargs)
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

    def get_preferred_backend(self) -> str:
        return "pennylane"
