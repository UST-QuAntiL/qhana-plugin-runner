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

import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional, Callable

from ..data_loading_circuits import TreeLoader
from ..utils import int_to_bitlist, bitlist_to_int
from ..q_arithmetic import cc_increment_register
from ..ccnot import adaptive_ccnot
from .qknn import QkNN
from ..amplitude_amplification import (
    exp_searching_amplitude_amplification,
    get_exp_search_aa_representative_circuit,
)
from ..check_wires import check_wires_uniqueness, check_num_wires

from collections import Counter


def x_state_to_one(wires: List[int], state: List[int]):
    for (wire, value) in zip(wires, state):
        if value == 0:
            qml.PauliX((wire,))


def oracle_state_circuit(
    data_wires: List[int],
    oracle_wire: int,
    ancilla_wires: List[int],
    unclean_wires: List[int],
    good_states: List[List[int]],
):
    """
    Given a list of 'good' states, this function flips the oracle bit, if the data register is in a 'good' state
    """
    for state in good_states:
        x_state_to_one(data_wires, state)
        adaptive_ccnot(data_wires, ancilla_wires, unclean_wires, oracle_wire)
        x_state_to_one(data_wires, state)


def calc_hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    return np.bitwise_xor(x, y).sum()


class BasheerHammingQkNN(QkNN):
    def __init__(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        k: int,
        train_wires: List[int],
        idx_wires: List[int],
        ancilla_wires: List[int],
        backend: qml.Device,
        exp_itr: int = 10,
        unclean_wires: List[int] = None,
    ):
        super(BasheerHammingQkNN, self).__init__(train_data, train_labels, k, backend)

        self.train_data = np.array(train_data, dtype=int)

        self.exp_itr = exp_itr
        self.num_train_data = self.train_data.shape[0]
        self.train_data = self.repeat_data_til_next_power_of_two(self.train_data)

        self.unclean_wires = [] if unclean_wires is None else unclean_wires

        self.train_wires = train_wires
        self.idx_wires = idx_wires
        self.ancilla_wires = ancilla_wires
        wire_types = ["train", "idx", "ancilla", "unclean"]
        # Details in method get_necessary_wires
        num_wires = [
            self.train_data.shape[1],
            np.ceil(np.log2(self.train_data.shape[0])),
            int(np.ceil(np.log2(self.train_data.shape[1]))) + 5,
        ]
        error_msgs = [
            "the points' dimensionality.",
            "ceil(log2(size of train_data)).",
            "ceil(log2(the points' dimensionality))) + 5.",
        ]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-1], num_wires, error_msgs)

        a_len = int(np.ceil(np.log2(self.train_data.shape[1]))) + 2
        # Ancilla wires are split as follows:
        # [0, a_len) are reserved for the overflow register
        # a_len is reserved as the oracle wire
        # a_len + 1 is reserved as the threshold wire
        # a_len + 2 is rserved as the not in list wire
        # Thus we need int(np.ceil(np.log2(self.train_data.shape[1]))) + 5 wires
        self.overflow_wires = self.ancilla_wires[
            :a_len
        ]  # This register is used with the threshold oracle by Ruan et al.
        self.oracle_wire = self.ancilla_wires[a_len]  # This is the oracles qubit
        self.threshold_wire = self.ancilla_wires[
            a_len + 1
        ]  # This is the thresholds oracle's qubit
        self.not_in_list_wire = self.ancilla_wires[
            a_len + 2
        ]  # This i the not_in_list oracle's qubit
        self.additional_ancilla_wires = self.ancilla_wires[
            a_len + 3 :
        ]  # Any additional wires

        self.tree_loader = TreeLoader(
            self.prepare_data_for_treeloader(self.train_data),
            self.idx_wires,
            self.train_wires,
            self.ancilla_wires,
            unclean_wires=unclean_wires,
        )

    def prepare_data_for_treeloader(self, data: np.ndarray) -> np.ndarray:
        """
        The TreeLoader used, to load in the trainings data into a quantum computer, needs the amplitudes of the quantum
        states. The information stored in data is not the amplitudes, but rather which qubit should be flipped to one
        and which should stay zero.
        """
        tree_points = np.zeros((data.shape[0], 2 ** data.shape[1]))
        for idx, point in enumerate(data):
            tree_points[idx][bitlist_to_int(point)] = 1
        return tree_points

    def repeat_data_til_next_power_of_two(self, data: np.ndarray) -> np.ndarray:
        next_power = 2 ** int(np.ceil(np.log2(data.shape[0])))
        missing_till_next_power = next_power - data.shape[0]
        return np.vstack((data, data[:missing_till_next_power]))

    def get_oracle_wire_to_one_circuit(
        self, x: np.ndarray, a: List[int], indices: List[int]
    ) -> Callable[[], None]:
        """
        Returns a quantum circuit that inverses the oracle qubit of a trainings point, if its hamming distance is smaller
        than a certain threshold (oracle by Ruan et al.) and if its index is not in the list of the 'k' chosen indices.
        """
        def circuit():
            # Load points into register
            self.tree_loader.circuit()

            # Get inverse Hamming Distance
            for x_, train_wire in zip(x, self.train_wires):
                if x_ == 0:
                    qml.PauliX((train_wire,))

            # Prep overflow register
            for a_, overflow_wire in zip(a, self.overflow_wires):
                if a_ == 1:
                    qml.PauliX((overflow_wire,))

            # Increment overflow register for each 1 in the train register
            qml.PauliX(
                (self.threshold_wire,)
            )  # Allows us to set indicator_is_zero to False
            for t_idx, t_wire in enumerate(self.train_wires):
                cc_increment_register(
                    [t_wire],
                    self.overflow_wires,
                    [self.not_in_list_wire, self.oracle_wire]
                    + self.additional_ancilla_wires,
                    self.threshold_wire,
                    unclean_wires=self.unclean_wires
                    + self.train_wires[:t_idx]
                    + self.train_wires[t_idx + 1 :]
                    + self.idx_wires,
                    indicator_is_zero=False,
                )

            for overflow_wire in self.overflow_wires[:2]:
                qml.PauliX((overflow_wire,))

            # Set ancilla wire len(a) to 1, if the distance is smaller than the threshold
            adaptive_ccnot(
                self.overflow_wires[:2],
                [self.oracle_wire, self.not_in_list_wire] + self.additional_ancilla_wires,
                self.train_wires,  # We know that self.train_wires suffice for an unclean ccnot
                self.threshold_wire,
            )
            # Normally, we would need to invert the threshold wire here, but we already did that above, for the
            # increment steps. Thus, there is no need for qml.PauliX((self.threshold_wire, ))

            # Set ancilla wire len(a)+1 to 1, if the point's idx is not contained in indices
            # We are looping elements in self.train_data. Thus, a point at index i might also be at the index i+self.num_train_data.
            # Temp is a list of indices + the looped indices.
            temp = []
            for idx in indices:
                if idx + self.num_train_data < len(self.train_data):
                    temp.append(idx + self.num_train_data)
            # Now we convert the indices to their bits.
            temp = np.append(indices, np.array(temp, dtype=int))
            temp = [int_to_bitlist(value, len(self.idx_wires)) for value in temp]

            # Inverse qubit len(a)+1 so that the oracle is inversed
            qml.PauliX((self.not_in_list_wire,))
            # Set qubit len(a)+1 to 0, if idx qubits are in the list temp
            oracle_state_circuit(
                self.idx_wires,
                self.not_in_list_wire,
                [self.oracle_wire] + self.additional_ancilla_wires,
                self.unclean_wires,
                temp,
            )

            # Set oracle wire to 1, if the distance is smaller than the threshold
            # and the point's idx is not contained in indices
            qml.Toffoli(
                wires=(self.threshold_wire, self.not_in_list_wire, self.oracle_wire)
            )

        return circuit

    def get_phase_oracle_circuit(
        self, x: np.ndarray, a: List[int], indices: List[int]
    ) -> Callable[[], None]:
        """
        Returns a quantum circuit that gives a trainings point a phase of -1, if its hamming distance is smaller
        than a certain threshold (oracle by Ruan et al.) and if its index is not in the list of the 'k' chosen indices.
        """
        oracle_circuit = self.get_oracle_wire_to_one_circuit(x, a, indices)

        def quantum_circuit():
            # We do not use HX oracle XH here, since the oracle has a lot to uncompute and it only saves one Toffoli operation
            # Set oracle wire to one
            oracle_circuit()

            # Set phase
            qml.PauliZ((self.oracle_wire,))

            # Uncompute oracle
            qml.adjoint(oracle_circuit)()

        return quantum_circuit

    def idx_circuit(self):
        """
        Initialises the index register with a Walsh-Hadamard transform
        """
        for wire in self.idx_wires:
            qml.Hadamard((wire,))

    def zero_circuit(self):
        """
        Quantum circuit that gives the index |0> a -1 phase.
        """
        # Check if idx_wires = |0>
        for wire in self.idx_wires:
            qml.PauliX((wire,))

        qml.PauliX((self.oracle_wire,))
        qml.Hadamard((self.oracle_wire,))
        adaptive_ccnot(
            self.idx_wires,
            self.additional_ancilla_wires + [self.not_in_list_wire] + [self.threshold_wire] + self.overflow_wires + self.train_wires,
            self.unclean_wires,
            self.oracle_wire,
        )
        qml.Hadamard((self.oracle_wire,))
        qml.PauliX((self.oracle_wire,))

        for wire in self.idx_wires:
            qml.PauliX((wire,))

    def get_better_training_point_idx(
        self, x: np.ndarray, distance_threshold: int, chosen_indices: List[int]
    ) -> Tuple[Optional[int], Optional[int]]:
        if distance_threshold < 0:
            return None, None
        state_circuit = self.idx_circuit
        zero_circuit = self.zero_circuit

        distance_threshold = min(distance_threshold, self.train_data.shape[0])
        p = int(np.ceil(np.log2(self.train_data.shape[1])))
        a = int(2**p - self.train_data.shape[1] + distance_threshold)
        a = int_to_bitlist(a, p + 2)

        oracle_phase_circuit = self.get_phase_oracle_circuit(x, a, chosen_indices)
        oracle_one_circuit = self.get_oracle_wire_to_one_circuit(x, a, chosen_indices)
        check_if_good_wire = self.oracle_wire
        measure_wires = self.idx_wires + self.train_wires

        result = exp_searching_amplitude_amplification(
            state_circuit,
            state_circuit,
            zero_circuit,
            oracle_phase_circuit,
            self.backend,
            oracle_one_circuit,
            check_if_good_wire,
            measure_wires,
            exp_itr=self.exp_itr,
        )

        if result is None:
            return None, None

        idx_bits = result[: len(self.idx_wires)]
        train_bits = result[len(self.idx_wires) :]
        hamming_distance = len(train_bits) - np.array(train_bits).sum()
        return int(bitlist_to_int(idx_bits) % self.num_train_data), hamming_distance

    def label_point(self, x: np.ndarray) -> int:
        x = np.array(x, dtype=int)
        # Init: First choose k random points, to be the current nearest neighbours
        chosen_indices = np.random.choice(
            range(self.num_train_data), self.k, replace=False
        )

        chosen_distances = np.array(
            [calc_hamming_distance(x, self.train_data[idx]) for idx in chosen_indices]
        )

        # Loop converges, if no better y can be found
        # Loop should take at most |train data| - k many steps
        for _ in range(self.num_train_data - self.k):
            # Choose index y from the indices in A
            # Choosing y with max distance to x is best
            y_idx = chosen_distances.argmax()

            # Find new_y with quantum algorithm such that new_y_distance < y_distance and new_y not in A
            new_y, distance = self.get_better_training_point_idx(
                x, chosen_distances[y_idx] - 1, chosen_indices
            )
            if new_y is not None and new_y != chosen_indices[y_idx]:
                # Replace y with new_y
                chosen_indices[y_idx] = new_y
                chosen_distances[y_idx] = distance
            else:
                # No closer neighbor has been found
                break

        # Majority voting
        counts = Counter(
            self.train_labels[chosen_indices]
        )  # Count occurrences of labels in k smallest values
        new_label = max(counts, key=counts.get)  # Get most frequent label
        return new_label

    @staticmethod
    def get_necessary_wires(train_data: np.ndarray) -> Tuple[int, int, int]:
        # train wires: we need a qubit for each dimension of a point
        # idx wires: a number n can be represented by log(n) bits. Thus, we need ceil(log2(size of train_data)) qubits
        # ancilla wires:    + we need int(np.ceil(np.log2(self.train_data.shape[1])))+2 qubits for the overflow register
        #                   + we need 1 qubit for the oracle
        #                   + we need 1 qubit for the threshold oracle
        #                   + we need 1 qubit for the not_in_list oracle
        return (
            train_data.shape[1],
            int(np.ceil(np.log2(train_data.shape[0]))),
            int(np.ceil(np.log2(train_data.shape[1]))) + 5,
        )

    def get_representative_circuit(self, X: np.ndarray) -> str:
        # Chose some initial parameters
        x = X[0]
        distance_threshold = 1
        chosen_indices = list(range(self.k))

        # Initiate all necessary circuits for the aa circuit
        state_circuit = self.idx_circuit
        zero_circuit = self.zero_circuit

        distance_threshold = min(distance_threshold, self.train_data.shape[0])
        p = int(np.ceil(np.log2(self.train_data.shape[1])))
        a = int(2**p - self.train_data.shape[1] + distance_threshold)
        a = int_to_bitlist(a, p + 2)

        oracle_phase_circuit = self.get_phase_oracle_circuit(x, a, chosen_indices)
        oracle_one_circuit = self.get_oracle_wire_to_one_circuit(x, a, chosen_indices)
        check_if_good_wire = self.oracle_wire
        measure_wires = self.idx_wires + self.train_wires

        # Get representative aa circuit
        circuit = get_exp_search_aa_representative_circuit(
            state_circuit,
            state_circuit,
            zero_circuit,
            oracle_phase_circuit,
            self.backend,
            oracle_one_circuit,
            check_if_good_wire,
            measure_wires,
        )
        circuit.construct([], {})
        return circuit.qtape.to_openqasm()

    def heatmap_meaningful(self) -> bool:
        return False
