import pennylane as qml
import numpy as np
from typing import List, Tuple

from ..data_loading_circuits import TreeLoader
from ..utils import int_to_bitlist, bitlist_to_int
from ..q_arithmetic import cc_increment_register
from ..ccnot import adaptive_ccnot
from .qknn import QkNN
from ..amplitude_amplification import exp_searching_amplitude_amplification, get_exp_search_aa_representative_circuit
from ..check_wires import check_wires_uniqueness, check_num_wires

from collections import Counter


def string_indices_and_distances(indices, distances, separator=" "):
    result = []
    for idx, d in zip(indices, distances):
        result.append(f"{idx}:{d}")
    return separator.join(result)


def x_state_to_one(wires, state):
    for (wire, value) in zip(wires, state):
        if value == 0:
            qml.PauliX((wire, ))


def oracle_state_circuit(data_wires, oracle_wire, ancilla_wires, unclean_wires, good_states):
    for state in good_states:
        x_state_to_one(data_wires, state)
        adaptive_ccnot(data_wires, ancilla_wires, unclean_wires, oracle_wire)
        x_state_to_one(data_wires, state)


def calc_hamming_distance(x, y):
    return np.bitwise_xor(x, y).sum()


class BasheerHammingQkNN(QkNN):
    def __init__(self, train_data, train_labels, k: int,
                 train_wires: List[int], idx_wires: List[int], ancilla_wires: List[int], backend: qml.Device, exp_itr=10, unclean_wires=None):
        super(BasheerHammingQkNN, self).__init__(train_data, train_labels, k, backend)

        self.train_data = np.array(train_data, dtype=int)

        # self.a = int_to_bitlist(self.a, int(np.ceil(np.log2(self.a)))+1)
        # self.log2_threshold = 0 if self.distance_threshold == 0 else int(np.ceil(np.log2(self.distance_threshold)))
        self.exp_itr = exp_itr
        self.num_train_data = self.train_data.shape[0]
        self.train_data = self.repeat_data_til_next_power_of_two(self.train_data)

        self.unclean_wires = [] if unclean_wires is None else unclean_wires

        self.train_wires = train_wires
        self.idx_wires = idx_wires
        self.ancilla_wires = ancilla_wires
        wire_types = ["train", "idx", "ancilla", "unclean"]
        # Details in method get_necessary_wires
        num_wires = [self.train_data.shape[1], np.ceil(np.log2(self.train_data.shape[0])), int(np.ceil(np.log2(self.train_data.shape[1]))) + 5]
        error_msgs = ["the points' dimensionality.", "ceil(log2(size of train_data)).", "ceil(log2(the points' dimensionality))) + 5."]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-1], num_wires, error_msgs)

        a_len = (int(np.ceil(np.log2(self.train_data.shape[1]))) + 2)
        # Ancilla wires are split as follows:
        # [0, a_len) are reserved for the overflow register
        # a_len is reserved as the oracle wire
        # a_len + 1 is reserved as the threshold wire
        # a_len + 2 is rserved as the not in list wire
        # Thus we need int(np.ceil(np.log2(self.train_data.shape[1]))) + 5 wires
        self.overflow_wires = self.ancilla_wires[:a_len]                # This register is used with the threshold oracle by Ruan et al.
        self.oracle_wire = self.ancilla_wires[a_len]                    # This is the oracles qubit
        self.threshold_wire = self.ancilla_wires[a_len+1]               # This is the thresholds oracle's qubit
        self.not_in_list_wire = self.ancilla_wires[a_len+2]             # This i the not_in_list oracle's qubit
        self.additional_ancilla_wires = self.ancilla_wires[a_len+3:]    # Any additional wires

        self.tree_loader = TreeLoader(
            self.prepare_data_for_treeloader(self.train_data), self.idx_wires, self.train_wires, self.ancilla_wires,
            unclean_wires=unclean_wires
        )

    def prepare_data_for_treeloader(self, data):
        tree_points = np.zeros(((data.shape[0], 2**data.shape[1])))
        for idx, point in enumerate(data):
            tree_points[idx][bitlist_to_int(point)] = 1
        return tree_points

    def repeat_data_til_next_power_of_two(self, data):
        next_power = 2 ** int(np.ceil(np.log2(data.shape[0])))
        missing_till_next_power = next_power - data.shape[0]
        data = np.vstack((data, data[:missing_till_next_power]))
        return data

    def get_oracle_wire_to_one_circuit(self, x, a, indices, do_final_toffoli=True):
        def circuit():
            # Load points into register
            self.tree_loader.circuit()

            # Get inverse Hamming Distance
            for i in range(len(x)):
                if x[i] == 0:
                    qml.PauliX((self.train_wires[i],))

            # Prep overflow register
            for i in range(len(a)):
                if a[i] == 1:
                    qml.PauliX((self.overflow_wires[i],))

            # Increment overflow register for each 1 in the train register
            qml.PauliX((self.oracle_wire,))  # Allows us to set indicator_is_zero to False
            for t_idx, t_wire in enumerate(self.train_wires):
                cc_increment_register(
                    [t_wire],
                    self.overflow_wires,
                    [self.not_in_list_wire, self.threshold_wire] + self.additional_ancilla_wires,
                    self.oracle_wire,
                    unclean_wires=self.unclean_wires + self.train_wires[:t_idx] + self.train_wires[t_idx+1:] + self.idx_wires,
                    indicator_is_zero=False,
                )
            qml.PauliX((self.oracle_wire,))  # Undo the inverse (indicator_is_zero thing)

            for i in range(2):
                qml.PauliX((self.overflow_wires[i],))

            # Set ancilla wire len(a) to 1, if the distance is smaller than the threshold
            adaptive_ccnot(
                self.overflow_wires[:2],
                [self.oracle_wire, self.not_in_list_wire] + self.additional_ancilla_wires,
                self.train_wires,                   # We know that self.train_wires suffice for an unclean ccnot
                self.threshold_wire
            )
            qml.PauliX((self.threshold_wire, ))

            # Set ancilla wire len(a)+1 to 1, if the point's idx is not contained in indices
            # We are looping elements in self.train_data. Thus, a point at index i might also be at the index i+self.num_train_data.
            # Temp is a list of indices + the looped indices.
            temp = []
            for idx in indices:
                if idx + self.num_train_data < len(self.train_data):
                    temp.append(idx+self.num_train_data)
            # Now we convert the indices to their bits.
            temp = np.append(indices, np.array(temp, dtype=int))
            temp = [int_to_bitlist(value, len(self.idx_wires)) for value in temp]

            # Inverse qubit len(a)+1 so that the oracle is inversed
            qml.PauliX((self.not_in_list_wire,))
            # Set qubit len(a)+1 to 0, if idx qubits are in the list temp
            oracle_state_circuit(
                self.idx_wires, self.not_in_list_wire,
                [self.oracle_wire] + self.additional_ancilla_wires,
                self.unclean_wires, temp
            )

            # Set oracle wire to 1, if the distance is smaller than the threshold
            # and the point's idx is not contained in indices
            qml.Toffoli(wires=(self.threshold_wire, self.not_in_list_wire, self.oracle_wire))

        return circuit

    def get_phase_oracle_circuit(self, x, a, indices):
        oracle_circuit = self.get_oracle_wire_to_one_circuit(x, a, indices)

        def quantum_circuit():
            # We do not use HX oracle XH here, since the oracle has a lot to uncompute and it only saves one Toffoli operation
            # Set oracle wire to one
            oracle_circuit()

            # Set phase
            qml.PauliZ((self.oracle_wire, ))

            # Uncompute oracle
            qml.adjoint(oracle_circuit)()

        return quantum_circuit

    def idx_circuit(self):
        for wire in self.idx_wires:
            qml.Hadamard((wire, ))

    def zero_circuit(self):
        # Check if idx_wires = |0>
        for wire in self.idx_wires:
            qml.PauliX((wire, ))

        qml.Hadamard((self.ancilla_wires[0], ))
        adaptive_ccnot(self.idx_wires, self.ancilla_wires[1:]+self.train_wires, self.unclean_wires, self.ancilla_wires[0])
        qml.Hadamard((self.ancilla_wires[0], ))

        for wire in self.idx_wires:
            qml.PauliX((wire, ))

    def get_better_training_point_idx(self, x, distance_threshold, chosen_indices) -> Tuple[int, int]:
        if distance_threshold < 0:
            return None, None
        state_circuit = self.idx_circuit
        zero_circuit = self.zero_circuit

        distance_threshold = min(distance_threshold, self.train_data.shape[0])
        p = int(np.ceil(np.log2(self.train_data.shape[1])))
        a = int(2 ** p - self.train_data.shape[1] + distance_threshold)
        a = int_to_bitlist(a, p + 2)

        oracle_phase_circuit = self.get_phase_oracle_circuit(x, a, chosen_indices)
        oracle_one_circuit = self.get_oracle_wire_to_one_circuit(x, a, chosen_indices)
        check_if_good_wire = self.oracle_wire
        measure_wires = self.idx_wires + self.train_wires

        result = exp_searching_amplitude_amplification(
            state_circuit, state_circuit, zero_circuit, oracle_phase_circuit, self.backend,
            oracle_one_circuit, check_if_good_wire, measure_wires, exp_itr=self.exp_itr
        )

        if result is None:
            return None, None

        idx_bits = result[:len(self.idx_wires)]
        train_bits = result[len(self.idx_wires):]
        hamming_distance = len(train_bits) - np.array(train_bits).sum()
        return int(bitlist_to_int(idx_bits) % self.num_train_data), hamming_distance

    def label_point(self, x) -> int:
        x = np.array(x, dtype=int)
        # Init: First choose k random points, to be the current nearest neighbours
        chosen_indices = np.random.choice(range(self.num_train_data), self.k, replace=False)

        # chosen_indices = np.array([0, 4])
        chosen_distances = np.array([calc_hamming_distance(x, self.train_data[idx]) for idx in chosen_indices])
        converged = False

        # Loop
        while not converged:
            # Choose index y from the indices in A
            # Choosing y with max distance to x is best
            y_idx = chosen_distances.argmax()

            # Find new_y with quantum algorithm such that new_y_distance < y_distance and new_y not in A
            new_y, distance = self.get_better_training_point_idx(x, chosen_distances[y_idx]-1, chosen_indices)
            if new_y is not None and new_y != chosen_indices[y_idx]:
                # Replace y with new_y
                chosen_indices[y_idx] = new_y
                chosen_distances[y_idx] = distance
            else:
                converged = True

        # Majority voting
        counts = Counter(self.train_labels[chosen_indices])  # Count occurrences of labels in k smallest values
        new_label = max(counts, key=counts.get)  # Get most frequent label
        return new_label

    @staticmethod
    def get_necessary_wires(train_data):
        # train wires: we need a qubit for each dimension of a point
        # idx wires: a number n can be represented by log(n) bits. Thus, we need ceil(log2(size of train_data)) qubits
        # ancilla wires:    + we need int(np.ceil(np.log2(self.train_data.shape[1])))+2 qubits for the overflow register
        #                   + we need 1 qubit for the oracle
        #                   + we need 1 qubit for the threshold oracle
        #                   + we need 1 qubit for the not_in_list oracle
        return train_data.shape[1], int(np.ceil(np.log2(train_data.shape[0]))), int(np.ceil(np.log2(train_data.shape[1]))) + 5

    def get_representative_circuit(self, X) -> str:
        # Chose some initial parameters
        x = X[0]
        distance_threshold = 1
        chosen_indices = range(self.k)

        # Initiate all necessary circuits for the aa circuit
        state_circuit = self.idx_circuit
        zero_circuit = self.zero_circuit

        distance_threshold = min(distance_threshold, self.train_data.shape[0])
        p = int(np.ceil(np.log2(self.train_data.shape[1])))
        a = int(2 ** p - self.train_data.shape[1] + distance_threshold)
        a = int_to_bitlist(a, p + 2)

        oracle_phase_circuit = self.get_phase_oracle_circuit(x, a, chosen_indices)
        oracle_one_circuit = self.get_oracle_wire_to_one_circuit(x, a, chosen_indices)
        check_if_good_wire = self.oracle_wire
        measure_wires = self.idx_wires + self.train_wires

        # Get representative aa circuit
        circuit = get_exp_search_aa_representative_circuit(
            state_circuit, state_circuit, zero_circuit, oracle_phase_circuit, self.backend,
            oracle_one_circuit, check_if_good_wire, measure_wires

        )
        circuit.construct([], {})
        return circuit.qtape.to_openqasm()