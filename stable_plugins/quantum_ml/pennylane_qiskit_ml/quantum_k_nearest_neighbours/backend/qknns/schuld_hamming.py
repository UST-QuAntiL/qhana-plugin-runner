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

from typing import List, Callable, Tuple
import numpy as np

import pennylane as qml
from ..data_loading_circuits import QAM
from .qknn import QkNN
from ..utils import (
    bitlist_to_int,
    int_to_bitlist,
    check_binary,
    ceil_log2,
)
from ..check_wires import check_wires_uniqueness, check_num_wires


class SchuldQkNN(QkNN):
    def __init__(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        idx_wires: List[int],
        train_wires: List[int],
        label_wires: List[int],
        qam_ancilla_wires: List[int],
        backend: qml.Device,
        unclean_wires: List[int] = None,
    ):
        super(SchuldQkNN, self).__init__(
            train_data, train_labels, len(train_data), backend
        )

        check_binary(
            self.train_data,
            "All the data needs to be binary, when dealing with the hamming distance",
        )

        self.train_data = np.array(train_data, dtype=int)

        self.label_indices = self.init_labels(train_labels)

        self.unclean_wires = [] if unclean_wires is None else unclean_wires

        self.idx_wires = idx_wires
        self.train_wires = train_wires
        self.qam_ancilla_wires = qam_ancilla_wires
        self.label_wires = label_wires
        wire_types = ["idx", "train", "label", "qam_ancilla", "unclean"]
        num_idx_wires = int(np.ceil(np.log2(self.train_data.shape[0])))
        num_wires = [
            num_idx_wires,
            self.train_data.shape[1],
            self.label_indices.shape[1],
            max(self.train_data.shape[1], 2),
        ]
        error_msgs = [
            "the points' dimensionality.",
            "ceil(log2(len(unique labels))).",
            "the points' dimensionality and greater or equal to 2.",
        ]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-1], num_wires, error_msgs)

        self.qam = QAM(
            np.array(
                [
                    int_to_bitlist(i, num_idx_wires)
                    for i in range(self.train_data.shape[0])
                ]
            ),  # The indices
            self.idx_wires,
            self.qam_ancilla_wires,
            additional_bits=np.concatenate((self.train_data, self.label_indices), axis=1),
            additional_wires=self.train_wires + self.label_wires,
            unclean_wires=unclean_wires,
        )

    def init_labels(self, labels: List[int]) -> np.ndarray:
        """
        This function maps the labels to their index in self.unique_labels
        """
        label_indices = []
        # Map labels to their index. The index is represented by a list of its bits
        label_to_idx = {}
        # Number of bits needed to represent all indices of our labels
        num_bits_needed = ceil_log2(len(self.unique_labels))
        for i, unique_label in enumerate(self.unique_labels):
            label_to_idx[unique_label] = int_to_bitlist(i, num_bits_needed)
        for label in labels:
            label_indices.append(label_to_idx[label])
        return np.array(label_indices)

    def get_label_from_samples(self, samples: List[List[int]]) -> int:
        """
        Given a list of samples, this function returns the label with the most occurrences, where an oracle qubit
        is equal to |0>.
        """
        label_probs = np.zeros(len(self.unique_labels))
        counts = np.zeros(len(self.unique_labels))
        for sample in samples:
            label = bitlist_to_int(sample[1:])
            if sample[0] == 0 and label < len(label_probs):
                label_probs[label] += 1
            if label < len(label_probs):
                counts[label] += 1
        return self.unique_labels[label_probs.argmax()]

    def get_quantum_circuit(self, x: np.ndarray) -> Callable[[], None]:
        """
        Returns a quantum circuit that does the following:
        1. Load in the trainings data with a quantum associative memory, i.e. initialise the label- and data-register
        2. Invert the i'th qubit of the data-register, if the i'th bit of the test point is 0
           => The sum of the register is the inverse hamming distance
        3. Rotate an oracle qubit more towards |1>, for each |1> in the data-register
        4. sample the oracle qubit and the label-register
        The oracle will be rotated at most m times, where m is the dimensionality of a trainings/test point.
        Thus, in step 3 the oracle qubit will be rotated by pi/m for each |1>.
        """
        rot_angle = np.pi / self.train_data.shape[1]

        @qml.qnode(self.backend)
        def circuit():
            self.qam.circuit()
            for x_, train_wire in zip(x, self.train_wires):
                if x_ == 1:
                    qml.PauliX((train_wire,))
            for train_wire in self.train_wires:
                # QAM ancilla wires are 0 after QAM -> use one of those wires
                qml.CRX(rot_angle, wires=(train_wire, self.qam_ancilla_wires[0]))
            return qml.sample(wires=[self.qam_ancilla_wires[0]] + self.label_wires)

        return circuit

    def label_point(self, x: np.ndarray) -> int:
        check_binary(
            x, "All the data needs to be binary, when dealing with the hamming distance"
        )
        samples = self.get_quantum_circuit(x)()
        return self.get_label_from_samples(samples)

    @staticmethod
    def get_necessary_wires(
        train_data: np.ndarray, train_labels: np.ndarray
    ) -> Tuple[int, int, int, int]:
        return (
            int(np.ceil(np.log2(train_data.shape[0]))),
            len(train_data[0]),
            ceil_log2(len(set(train_labels))),
            max(len(train_data[0]), 2),
        )

    def get_representative_circuit(self, X: np.ndarray) -> str:
        circuit = self.get_quantum_circuit(X[0])
        circuit.construct([], {})
        return circuit.qtape.to_openqasm()

    def heatmap_meaningful(self) -> bool:
        return False
