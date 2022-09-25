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

    def check_wires(self, wire_types):
        for idx1 in range(len(wire_types)):
            wire_type1 = wire_types[idx1]
            wires1 = getattr(self, wire_type1 + '_wires')
            for idx2 in range(idx1 + 1, len(wire_types)):
                wire_type2 = wire_types[idx2]
                wires2 = getattr(self, wire_type2 + '_wires')
                if any(wire in wires1 for wire in wires2):
                    raise QuantumFunctionError(
                        f"The {wire_type1} wires must be different from the {wire_type2} wires"
                    )

    def check_num_wires(self, wire_types, num_wires, error_msgs):
        for i in range(len(wire_types)):
            wire_type = wire_types[i]
            wires = getattr(self, wire_type + '_wires')
            if len(wires) < num_wires[i]:
                error = f"The number of {wire_type} wires has to be greater or equal to {error_msgs[i]}"
                raise QuantumFunctionError(error)
