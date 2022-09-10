from .qknn import QkNN
import pennylane as qml
from abc import abstractmethod
from typing import List
import numpy as np
from ..data_loading_circuits.quantum_associative_memory import QAM
from ..utils import bitlist_to_int, check_if_values_are_binary
from collections import Counter


from celery.utils.log import get_task_logger
TASK_LOGGER = get_task_logger(__name__)

class SimpleQkNN(QkNN):
    def __init__(self, train_data, train_labels, k: int, backend: qml.Device):
        super(SimpleQkNN, self).__init__(train_data, train_labels, k, backend, 0)

    @abstractmethod
    def calculate_distances(self, x) -> List[float]:
        """
        Calculates and returns the distances for each point to x.
        :param x:
        :return:
        """

    def label_point(self, x):
        if self.k == len(self.train_data):
            new_label = np.bincount(self.train_labels).argmax()
        else:
            distances = np.array(self.calculate_distances(x))       # Get distances
            indices = np.argpartition(distances, self.k)[:self.k]   # Get k smallest values
            counts = Counter(self.train_labels[indices])            # Count occurrences of labels in k smallest values
            TASK_LOGGER.info(f"counts = {counts}")
            new_label = max(counts, key=counts.get)                 # Get most frequent label
        return new_label


class SimpleHammingQkNN(SimpleQkNN):
    def __init__(self, train_data, train_labels, k: int,
                 train_wires: List[int], qam_ancilla_wires: List[int], backend: qml.Device):
        super(SimpleHammingQkNN, self).__init__(train_data, train_labels, k, backend)
        self.train_data = np.array(train_data, dtype=int)

        if not check_if_values_are_binary(self.train_data):
            raise ValueError("All the data needs to be binary, when dealing with the hamming distance")

        self.point_num_to_idx = dict()
        for i in range(len(self.train_data)):
            self.point_num_to_idx[bitlist_to_int(self.train_data[i])] = i

        self.train_wires = train_wires
        self.qam_ancilla_wires = qam_ancilla_wires
        wire_types = ['train', 'qam_ancilla']
        num_wires = [self.train_data.shape[1], self.train_data.shape[1]]
        error_msgs = ["the points' dimensionality.", "the points' dimensionality."]
        self.check_wires(wire_types)
        self.check_num_wires(wire_types, num_wires, error_msgs)
        self.qam = QAM(self.train_data, train_wires, qam_ancilla_wires[:2], qam_ancilla_wires[2:])

    def get_quantum_circuit(self, x, rot_angle):
        def quantum_circuit():
            self.qam.circuit()
            for i in range(len(x)):
                if x[i] == 0:
                    qml.PauliX((self.train_wires[i],))
            for i in range(len(self.train_wires)):
                # QAM ancilla wires are 0 after QAM -> use one of those wires
                qml.CRX(rot_angle, wires=(self.train_wires[i], self.qam_ancilla_wires[0]))
            for i in range(len(x)):
                if x[i] == 0:
                    qml.PauliX((self.train_wires[i],))
            return qml.sample(wires=self.train_wires + [self.qam_ancilla_wires[0]])
        return quantum_circuit

    def calculate_distances(self, x) -> List[float]:
        rot_angle = np.pi / self.train_data.shape[1]
        samples = qml.QNode(self.get_quantum_circuit(x, rot_angle), self.backend)().tolist()
        num_zero_ancilla = [0]*len(self.train_data)
        total_ancilla = [0]*len(self.train_data)
        # Count how often a certain point was measured (total_ancilla)
        # and how often the ancilla qubit was zero (num_zero_ancilla)
        for sample in samples:
            idx = self.point_num_to_idx[bitlist_to_int(sample[:-1])]
            total_ancilla[idx] += 1
            if sample[-1] == 0:
                num_zero_ancilla[idx] += 1
        # Get prob for ancilla qubit to be equal to 0
        for i in range(len(num_zero_ancilla)):
            # 0 <= num_zero_ancilla[i] / total_ancilla[i] <= 1. Hence if total_ancilla[i] == 0, we have no information
            # about the distance, but due to the rot_angle it can't be greater than 1, therefore we set it to 1
            num_zero_ancilla[i] = 1 if total_ancilla[i] == 0 else num_zero_ancilla[i] / total_ancilla[i]
        return num_zero_ancilla

    @staticmethod
    def get_necessary_wires(train_data):
        return len(train_data[0]), len(train_data[0])
