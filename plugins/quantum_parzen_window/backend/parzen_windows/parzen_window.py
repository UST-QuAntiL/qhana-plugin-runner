import numpy as np
from abc import abstractmethod
from enum import Enum
from typing import List
from pennylane import QuantumFunctionError, Device


class QParzenWindowEnum(Enum):
    ruan_window = "ruan et al. parzen window"

    def get_parzen_window(self, train_data, train_labels, window_size, backend, shots):
        if self == QParzenWindowEnum.ruan_window:
            from .ruan_parzen_window import RuanParzenWindow
            wires = self.check_and_get_qubits(RuanParzenWindow, backend, train_data=train_data, train_labels=train_labels)
            return RuanParzenWindow(train_data, train_labels, window_size, wires[0], wires[1], wires[2], backend)

    def check_and_get_qubits(self, qknn_class, backend, **kwargs):
        num_necessary_wires = qknn_class.get_necessary_wires(**kwargs)
        num_total_wires = 0
        wires = []
        for num_wires in num_necessary_wires:
            wires.append(list(range(num_total_wires, num_total_wires + num_wires)))
            num_total_wires += num_wires
        if num_total_wires > backend.num_wires:
            raise ValueError(
                "The quantum circuit needs at least " + str(num_total_wires)
                + " qubits, but it only got " + str(backend.num_wires) + "!"
            )
        return wires

    def get_preferred_backend(self):
        # if self == QkNNEnum.qiskit_qknn:
        #     return "qiskit"
        # else:
        #     return "pennylane"
        return "pennylane"


class ParzenWindow:
    def __init__(self, train_data, train_labels, distance_threshold: float, backend: Device):
        self.backend = backend
        if not isinstance(train_data, np.ndarray):
            train_data = np.array(train_data, dtype=int)
        self.train_data = train_data
        self.train_labels = train_labels
        self.unique_labels = list(set(self.train_labels))
        self.distance_threshold = distance_threshold

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
