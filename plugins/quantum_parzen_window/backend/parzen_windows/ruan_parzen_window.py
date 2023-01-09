from .parzen_window import ParzenWindow
from typing import List
import numpy as np
import pennylane as qml
from ..data_loading_circuits.quantum_associative_memory import QAM
from ..ccnot import adaptive_ccnot
from ..utils import int_to_bitlist, bitlist_to_int, check_if_values_are_binary
from ..q_arithmetic import cc_increment_register
from ..check_wires import check_wires_uniqueness, check_num_wires


class RuanParzenWindow(ParzenWindow):
    def __init__(
        self,
        train_data,
        train_labels,
        distance_threshold: float,
        train_wires: List[int],
        label_wires: List[int],
        ancilla_wires: List[int],
        backend: qml.Device,
        unclean_wires=None,
    ):
        super(RuanParzenWindow, self).__init__(
            train_data, train_labels, distance_threshold, backend
        )
        self.train_data = np.array(train_data, dtype=int)

        if not check_if_values_are_binary(self.train_data):
            raise ValueError(
                "All the data needs to be binary, when dealing with the hamming distance"
            )

        self.distance_threshold = min(
            int(self.distance_threshold), self.train_data.shape[1]
        )
        self.k = int(np.ceil(np.log2(self.train_data.shape[1])))
        self.a = int(2 ** self.k - self.train_data.shape[1] + self.distance_threshold)
        self.a = int_to_bitlist(self.a, self.k + 2)
        self.label_indices = self.init_labels(train_labels)

        self.unclean_wires = [] if unclean_wires is None else unclean_wires
        self.train_wires = train_wires
        self.label_wires = label_wires
        self.ancilla_wires = ancilla_wires

        wire_types = ["train", "label", "ancilla", "unclean"]
        num_wires = [
            self.train_data.shape[1],
            max(1, int(np.ceil(np.log2(len(self.unique_labels))))),
            np.ceil(np.log2(self.train_data.shape[1])) + 4,
        ]
        error_msgs = [
            "the points' dimensionality.",
            "ceil(log2(the points' dimensionality)))+2.",
            "ceil(log2(len(unique labels))).",
        ]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-1], num_wires, error_msgs)

        # Ancilla wires are split as follows:
        # [0, len(a)) are reserved for the overflow register
        # len(a) is reserved as the oracle wire
        # len(a) + 1 is an ancilla wire for ccnots
        # Thus we need len(a) + 2 = int(np.ceil(np.log2(self.train_data.shape[1]))) + 4 wires
        self.overflow_wires = self.ancilla_wires[
            : len(self.a)
        ]  # This register is used with the threshold oracle by Ruan et al.
        self.oracle_wire = self.ancilla_wires[len(self.a)]  # This is the oracles qubit
        self.additional_ancilla_wires = self.ancilla_wires[
            len(self.a) + 1 :
        ]  # Any additional wires

        self.qam = QAM(
            self.train_data,
            self.train_wires,
            self.ancilla_wires,
            additional_bits=self.label_indices,
            additional_wires=self.label_wires,
            unclean_wires=self.unclean_wires,
        )

    def init_labels(self, labels):
        label_indices = list()
        label_to_idx = (
            dict()
        )  # Map labels to their index. The index is represented by a list of its bits
        num_bits_needed = max(
            1, int(np.ceil(np.log2(len(self.unique_labels))))
        )  # Number of bits needed to represent all indices of our labels
        for i in range(len(self.unique_labels)):
            label_to_idx[self.unique_labels[i]] = int_to_bitlist(i, num_bits_needed)
        for label in labels:
            label_indices.append(label_to_idx[label])
        return np.array(label_indices)

    def get_quantum_circuit(self, x):
        def quantum_circuit():
            # Load points into register
            self.qam.circuit()

            # Get inverse Hamming Distance
            for i in range(len(x)):
                if x[i] == 0:
                    qml.PauliX((self.train_wires[i],))

            # Prep overflow register
            for i in range(len(self.a)):
                if self.a[i] == 1:
                    qml.PauliX((self.overflow_wires[i],))

            # Increment overflow register for each 1 in the train register
            qml.PauliX((self.oracle_wire,))  # Allows us to set indicator_is_zero to False
            for t_idx, t_wire in enumerate(self.train_wires):
                cc_increment_register(
                    [t_wire],
                    self.overflow_wires,
                    self.additional_ancilla_wires,
                    self.oracle_wire,
                    unclean_wires=self.unclean_wires
                    + self.train_wires[:t_idx]
                    + self.train_wires[t_idx + 1 :],
                    indicator_is_zero=False,
                )

            for i in range(2):
                qml.PauliX((self.overflow_wires[i],))
            adaptive_ccnot(
                self.overflow_wires[:2],
                self.additional_ancilla_wires,
                self.train_wires + self.unclean_wires,
                self.oracle_wire,
            )
            return qml.sample(wires=self.label_wires + [self.oracle_wire])

        return quantum_circuit

    def get_label_from_samples(self, samples):
        label_probs = np.zeros((len(self.unique_labels),))
        samples_with_one = 0
        for sample in samples:
            if sample[-1] == 1:
                label = bitlist_to_int(sample[:-1])
                label_probs[label] += 1
                samples_with_one += 1
        return self.unique_labels[label_probs.argmax()]

    def label_point(self, x) -> int:
        samples = qml.QNode(self.get_quantum_circuit(x), self.backend)().tolist()
        return self.get_label_from_samples(samples)

    @staticmethod
    def get_necessary_wires(train_data, train_labels):
        unique_labels = list(set(train_labels))
        return (
            int(len(train_data[0])),
            max(1, int(np.ceil(np.log2(len(unique_labels))))),
            int(np.ceil(np.log2(len(train_data[0]))) + 4),
        )

    def get_representative_circuit(self, X) -> str:
        circuit = qml.QNode(self.get_quantum_circuit(X[0]), self.backend)
        circuit.construct([], {})
        return circuit.qtape.to_openqasm()
