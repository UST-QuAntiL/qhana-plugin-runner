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

from .parzen_window import ParzenWindow
from typing import List, Callable, Tuple
import numpy as np
import pennylane as qml
from ..data_loading_circuits.quantum_associative_memory import QAM
from ..ccnot import adaptive_ccnot
from ..utils import int_to_bitlist, bitlist_to_int, is_binary, ceil_log2
from ..q_arithmetic import cc_increment_register
from ..check_wires import check_wires_uniqueness, check_num_wires


class RuanParzenWindow(ParzenWindow):
    def __init__(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        distance_threshold: float,
        idx_wires: List[int],
        train_wires: List[int],
        label_wires: List[int],
        ancilla_wires: List[int],
        backend: qml.Device,
        unclean_wires: List[int] = None,
    ):
        super(RuanParzenWindow, self).__init__(
            train_data, train_labels, distance_threshold, backend
        )
        self.train_data = np.array(train_data, dtype=int)

        if not is_binary(self.train_data):
            raise ValueError(
                "All the data needs to be binary, when dealing with the hamming distance"
            )

        self.distance_threshold = min(
            int(self.distance_threshold), self.train_data.shape[1]
        )
        self.k = ceil_log2(self.train_data.shape[1])
        self.a = int(2**self.k - self.train_data.shape[1] + self.distance_threshold)
        self.a = int_to_bitlist(self.a, self.k + 2)
        self.label_indices = self.init_labels(train_labels)

        self.unclean_wires = [] if unclean_wires is None else unclean_wires
        self.idx_wires = idx_wires
        self.train_wires = train_wires
        self.label_wires = label_wires
        self.ancilla_wires = ancilla_wires

        wire_types = ["idx", "train", "label", "ancilla", "unclean"]
        num_idx_wires = int(np.ceil(np.log2(self.train_data.shape[0])))
        num_wires = [
            num_idx_wires,
            self.train_data.shape[1],
            max(1, ceil_log2(len(self.unique_labels))),
            ceil_log2(self.train_data.shape[1]) + 4,
        ]
        error_msgs = [
            "the round up log2 of the number of points, i.e. ceil(log2(no. points)).",
            "the points' dimensionality.",
            "ceil(log2(no. of unique labels)) and greater or equal to 1.",
            "ceil(log2(the points' dimensionality)) + 4.",
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
            np.array(
                [
                    int_to_bitlist(i, num_idx_wires)
                    for i in range(self.train_data.shape[0])
                ]
            ),  # The indices
            self.idx_wires,
            self.ancilla_wires,
            additional_bits=np.concatenate((self.train_data, self.label_indices), axis=1),
            additional_wires=self.train_wires + self.label_wires,
            unclean_wires=self.unclean_wires,
        )

    def init_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        This function maps the labels to their index in self.unique_labels
        """
        label_indices = []
        # Map labels to their index. The index is represented by a list of its bits
        label_to_idx = {}
        # Number of bits needed to represent all indices of our labels
        num_bits_needed = max(1, ceil_log2(len(self.unique_labels)))
        for i, label in enumerate(self.unique_labels):
            label_to_idx[label] = int_to_bitlist(i, num_bits_needed)
        for label in labels:
            label_indices.append(label_to_idx[label])
        return np.array(label_indices)

    def get_quantum_circuit(self, x: np.ndarray) -> Callable[[], None]:
        """
        Returns a quantum circuit that does the following:
        1. Load in the trainings data with a quantum associative memory, i.e. initialise the label- and data-register
        2. Execute the oracle by Ruan, Y., Xue, X., Liu, H. et al. Quantum Algorithm for K-Nearest Neighbors Classification Based on the Metric of Hamming Distance. Int J Theor Phys 56, 3496–3507 (2017). https://doi.org/10.1007/s10773-017-3514-4
        """

        def quantum_circuit():
            # Load points into register
            self.qam.circuit()

            # Get inverse Hamming Distance
            for x_, train_wire in zip(x, self.train_wires):
                if x_ == 0:
                    qml.PauliX((train_wire,))

            # Prep overflow register
            for a_, overflow_wire in zip(self.a, self.overflow_wires):
                if a_ == 1:
                    qml.PauliX((overflow_wire,))

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

            for overflow_wire in self.overflow_wires[:2]:
                qml.PauliX((overflow_wire,))
            adaptive_ccnot(
                self.overflow_wires[:2],
                self.additional_ancilla_wires,
                self.train_wires + self.unclean_wires,
                self.oracle_wire,
            )
            return qml.sample(wires=self.label_wires + [self.oracle_wire])

        return quantum_circuit

    def get_label_from_samples(self, samples: List[List[int]]) -> int:
        """
        Given a list of samples, this function returns the label with the most occurrences, where an oracle qubit
        is equal to |1>.
        """
        label_probs = np.zeros(len(self.unique_labels))
        samples_with_one = 0
        for sample in samples:
            if sample[-1] == 1:
                label = bitlist_to_int(sample[:-1])
                if label < len(label_probs):
                    label_probs[label] += 1
                    samples_with_one += 1
        return self.unique_labels[label_probs.argmax()]

    def label_point(self, x: np.ndarray) -> int:
        samples = qml.QNode(self.get_quantum_circuit(x), self.backend)().tolist()
        return self.get_label_from_samples(samples)

    @staticmethod
    def get_necessary_wires(
        train_data: np.ndarray, train_labels: np.ndarray
    ) -> Tuple[int, int, int, int]:
        unique_labels = list(set(train_labels))
        return (
            int(np.ceil(np.log2(train_data.shape[0]))),
            int(len(train_data[0])),
            max(1, ceil_log2(len(unique_labels))),
            ceil_log2(len(train_data[0])) + 4,
        )

    def get_representative_circuit(self, X: np.ndarray) -> str:
        circuit = qml.QNode(self.get_quantum_circuit(X[0]), self.backend)
        circuit.construct([], {})
        return circuit.qtape.to_openqasm()
