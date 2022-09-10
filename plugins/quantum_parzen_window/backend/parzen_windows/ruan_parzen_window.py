from .parzen_window import ParzenWindow
from typing import List
import numpy as np
import pennylane as qml
from ..data_loading_circuits.quantum_associative_memory import QAM
from ..ccnot import unclean_ccnot
from ..utils import int_to_bitlist, bitlist_to_int, check_if_values_are_binary
from ..q_arithmetic import cc_increment_register


class RuanParzenWindow(ParzenWindow):
    def __init__(self, train_data, train_labels, distance_threshold: float,
                 train_wires: List[int], label_wires: List[int], qam_ancilla_wires: List[int], backend: qml.Device):
        super(RuanParzenWindow, self).__init__(train_data, train_labels, distance_threshold, backend)
        self.train_data = np.array(train_data, dtype=int)

        if not check_if_values_are_binary(self.train_data):
            raise ValueError("All the data needs to be binary, when dealing with the hamming distance")

        self.distance_threshold = min(int(self.distance_threshold), self.train_data.shape[1])
        self.k = int(np.ceil(np.log2(self.train_data.shape[1])))
        self.a = int(2**self.k - self.train_data.shape[1] + self.distance_threshold)
        self.a = int_to_bitlist(self.a, self.k+2)
        # self.a = int_to_bitlist(self.a, int(np.ceil(np.log2(self.a)))+1)
        # self.log2_threshold = 0 if self.distance_threshold == 0 else int(np.ceil(np.log2(self.distance_threshold)))
        self.label_indices = self.init_labels(train_labels)

        self.train_wires = train_wires
        self.label_wires = label_wires
        self.qam_ancilla_wires = qam_ancilla_wires
        wire_types = ['train', 'qam_ancilla', 'label']
        num_wires = [self.train_data.shape[1], self.train_data.shape[1], max(1, int(np.ceil(np.log2(len(self.unique_labels)))))]
        error_msgs = ["the points' dimensionality.", "the points' dimensionality.", "ceil(log2(len(unique labels)))."]
        self.check_wires(wire_types)
        self.check_num_wires(wire_types, num_wires, error_msgs)

        self.qam = QAM(
            self.train_data, self.train_wires, self.qam_ancilla_wires[:2], self.qam_ancilla_wires[2:],
            additional_bits=self.label_indices, additional_wires=self.label_wires
        )

    def init_labels(self, labels):
        label_indices = list()
        label_to_idx = dict()   # Map labels to their index. The index is represented by a list of its bits
        num_bits_needed = max(1, int(np.ceil(np.log2(len(self.unique_labels)))))    # Number of bits needed to represent all indices of our labels
        for i in range(len(self.unique_labels)):
            label_to_idx[self.unique_labels[i]] = int_to_bitlist(i, num_bits_needed)
        for label in labels:
            label_indices.append(label_to_idx[label])
        return np.array(label_indices)

    def get_quantum_circuit(self, x):
        def quantum_circuit():
            self.qam.circuit()  # Load points into register
            # Get inverse Hamming Distance
            for i in range(len(x)):
                if x[i] == 0:
                    qml.PauliX((self.train_wires[i],))
            # Prep overflow register
            for i in range(len(self.a)):
                if self.a[i] == 1:
                    qml.PauliX((self.qam_ancilla_wires[i],))
            # Increment overflow register for each 1 in the train register
            qml.PauliX((self.qam_ancilla_wires[2*len(self.a)],))    # Allows us to set ancilla_is_zero to False
            for i in range(len(self.train_wires)):
                cc_increment_register(
                    [self.train_wires[i]], self.qam_ancilla_wires[:len(self.a)],
                    self.qam_ancilla_wires[len(self.a):2*len(self.a)],
                    self.qam_ancilla_wires[2*len(self.a)], ancilla_is_zero=False
                )
            # for i in range(self.log2_threshold):
            for i in range(2):
                qml.PauliX((self.qam_ancilla_wires[:len(self.a)][i],))
            unclean_ccnot(self.qam_ancilla_wires[:len(self.a)][:2], self.train_wires, self.qam_ancilla_wires[2*len(self.a)])
            return qml.sample(wires=self.label_wires + [self.qam_ancilla_wires[2*len(self.a)]])
        return quantum_circuit

    def get_label_from_samples(self, samples):
        label_probs = np.zeros((len(self.unique_labels),))
        samples_with_one = 0
        for sample in samples:
            if sample[-1] == 1:
                label = bitlist_to_int(sample[:-1])
                label_probs[label] += 1
                samples_with_one += 1
        # if samples_with_one != 0:
        #     print(label_probs / samples_with_one)
        return self.unique_labels[label_probs.argmax()]

    def label_point(self, x) -> int:
        samples = qml.QNode(self.get_quantum_circuit(x), self.backend)().tolist()
        return self.get_label_from_samples(samples)

    @staticmethod
    def get_necessary_wires(train_data, train_labels):
        if len(train_data[0]) < 13:
            k = int(np.ceil(np.log2(len(train_data[0]))))
            return len(train_data[0]), int(np.ceil(np.log2(len(set(train_labels))))), 2*k + 5
        return len(train_data[0]), int(np.ceil(np.log2(len(set(train_labels))))), len(train_data[0])
