import numpy as np
from abc import abstractmethod
from enum import Enum
from typing import List
from pennylane import Device


def count_wires(wires: List[List]):
    s = 0
    for w in wires:
        s += len(w)
    return s


class QParzenWindowEnum(Enum):
    ruan_window = "ruan parzen window"

    def get_parzen_window(
        self, train_data, train_labels, window_size, max_wires, use_access_wires=True
    ):
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

    def check_and_get_qubits(self, qknn_class, max_wires, **kwargs):
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

    def get_preferred_backend(self):
        return "pennylane"


class ParzenWindow:
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
