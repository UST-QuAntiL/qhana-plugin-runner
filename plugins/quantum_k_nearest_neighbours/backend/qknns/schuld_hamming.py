import pennylane as qml
from ..data_loading_circuits.quantum_associative_memory import QAM
from .qknn import QkNN
from typing import List
import numpy as np
from ..utils import bitlist_to_int, int_to_bitlist, check_if_values_are_binary


from celery.utils.log import get_task_logger
TASK_LOGGER = get_task_logger(__name__)

import re


def num_decimal_places(value):
    TASK_LOGGER.info(f"num_decimal_places: value = {value} with type = {type(value)}")
    result = 0
    if value.startswith('-'):
        value = value[1:]
        result = 1
    m = re.match(r"^[0-9]*\.([1-9]([0-9]*[1-9])?)0*$", value)
    result += len(m.group(1)) if m is not None else 0
    TASK_LOGGER.info(f"num_decimal_places: result = {result}")
    return result


def simple_float_to_int(X):
    max_num_decimal_places = [0]*X.shape[1]
    for vec in X:
        for dim in range(X.shape[1]):
            max_num_decimal_places[dim] = max(num_decimal_places(str(vec[dim])), max_num_decimal_places[dim])
    max_num_bits = [0]*X.shape[1]
    for vec in X:
        for dim in range(X.shape[1]):
            temp = vec[dim]*(10**max_num_decimal_places[dim])
            if temp < 0:
                temp *= -1
            temp = int(np.ceil(np.log2(temp)))
            max_num_bits[dim] = max(temp, max_num_bits[dim])
    result = []
    for vec in X:
        new_vec = []
        for dim in range(X.shape[1]):
            new_vec += int_to_bitlist(int(vec[dim] * max_num_decimal_places[dim]), max_num_bits[dim])
        result.append(new_vec)
    return np.array(result)


class SchuldQkNN(QkNN):
    def __init__(self, train_data, train_labels, train_wires: List[int], label_wires: List[int], qam_ancilla_wires: List[int], backend: qml.Device):
        super(SchuldQkNN, self).__init__(train_data, train_labels, len(train_data), backend, 0)
        TASK_LOGGER.info(f"first point {train_data[0]}")
        self.train_data = np.array(train_data, dtype=int)

        if not check_if_values_are_binary(self.train_data):
            raise ValueError("All the data needs to be binary, when dealing with the hamming distance")

        self.label_indices = self.init_labels(train_labels)

        self.train_wires = train_wires
        self.qam_ancilla_wires = qam_ancilla_wires
        self.label_wires = label_wires
        wire_types = ['train', 'qam_ancilla', 'label']
        num_wires = [self.train_data.shape[1], self.train_data.shape[1], self.label_indices.shape[1]]
        error_msgs = ["the points' dimensionality.", "the points' dimensionality.", "ceil(log2(len(unique labels)))."]
        self.check_wires(wire_types)
        self.check_num_wires(wire_types, num_wires, error_msgs)

        self.qam = QAM(
            self.train_data, train_wires, qam_ancilla_wires[:2], qam_ancilla_wires[2:],
            additional_bits=self.label_indices, additional_wires=self.label_wires
        )

    def init_labels(self, labels):
        label_indices = list()
        label_to_idx = dict()  # Map labels to their index. The index is represented by a list of its bits
        num_bits_needed = int(
            np.ceil(np.log2(len(self.unique_labels))))  # Number of bits needed to represent all indices of our labels
        for i in range(len(self.unique_labels)):
            label_to_idx[self.unique_labels[i]] = int_to_bitlist(i, num_bits_needed)
        for label in labels:
            label_indices.append(label_to_idx[label])
        return np.array(label_indices)

    def get_label_from_samples(self, samples):
        label_probs = np.zeros((len(self.unique_labels),))
        for sample in samples:
            label = bitlist_to_int(sample[1:])
            if sample[0] == 0:
                label_probs[label] += 1
        print(label_probs / len(samples))
        return self.unique_labels[label_probs.argmax()]

    def get_circuit_results(self, x):
        rot_angle = np.pi / self.train_data.shape[1]
        @qml.qnode(self.backend)
        def circuit():
            self.qam.circuit()
            for i in range(len(x)):
                if x[i] == 0:
                    qml.PauliX((self.train_wires[i],))
            for i in range(len(self.train_wires)):
                # QAM ancilla wires are 0 after QAM -> use one of those wires
                qml.CRX(rot_angle, wires=(self.train_wires[i], self.qam_ancilla_wires[0]))
            return qml.sample(wires=[self.qam_ancilla_wires[0]] + self.label_wires)
        return circuit()

    def label_point(self, x):
        samples = self.get_circuit_results(x)
        return self.get_label_from_samples(samples)

    @staticmethod
    def get_necessary_wires(train_data, train_labels):
        return len(train_data[0]), int(np.ceil(np.log2(len(set(train_labels))))), len(train_data[0])
