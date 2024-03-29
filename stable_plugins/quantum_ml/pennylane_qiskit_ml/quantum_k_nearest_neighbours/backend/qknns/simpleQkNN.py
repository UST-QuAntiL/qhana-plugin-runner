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
from collections import Counter
from abc import abstractmethod
from typing import List, Tuple

from .qknn import QkNN
from ..data_loading_circuits import QAM
from ..data_loading_circuits import TreeLoader
from ..utils import int_to_bitlist, bitlist_to_int, check_binary, ceil_log2
from ..check_wires import check_wires_uniqueness, check_num_wires


class SimpleQkNN(QkNN):
    def __init__(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        k: int,
        backend: qml.Device,
    ):
        super(SimpleQkNN, self).__init__(train_data, train_labels, k, backend)

    @abstractmethod
    def calculate_distances(self, x: np.ndarray) -> List[float]:
        """
        Calculates and returns the distances for each point to x.
        :param x:
        :return:
        """

    def label_point(self, x: np.ndarray) -> int:
        """
        Computes the distances from the test point to each training point and assigns the test point with the most
        occurring label within the set of the k closest training points.
        """
        if self.k == len(self.train_data):
            new_label = np.bincount(self.train_labels).argmax()
        else:
            distances = np.array(self.calculate_distances(x))  # Get distances
            indices = np.argpartition(distances, self.k)[
                : self.k
            ]  # Get k smallest values
            counts = Counter(
                self.train_labels[indices]
            )  # Count occurrences of labels in k smallest values
            new_label = max(counts, key=counts.get)  # Get most frequent label
        return new_label


class SimpleHammingQkNN(SimpleQkNN):
    def __init__(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        k: int,
        idx_wires: List[int],
        train_wires: List[int],
        qam_ancilla_wires: List[int],
        backend: qml.Device,
        unclean_wires: List[int] = None,
    ):
        super(SimpleHammingQkNN, self).__init__(train_data, train_labels, k, backend)

        check_binary(
            self.train_data,
            "All the data needs to be binary, when dealing with the hamming distance",
        )
        self.train_data = np.array(train_data, dtype=int)

        self.unclean_wires = [] if unclean_wires is None else unclean_wires
        self.idx_wires = idx_wires
        self.train_wires = train_wires
        self.qam_ancilla_wires = qam_ancilla_wires
        wire_types = ["idx", "train", "qam_ancilla", "unclean"]
        num_idx_wires = int(np.ceil(np.log2(self.train_data.shape[0])))
        num_wires = [
            num_idx_wires,
            self.train_data.shape[1],
            max(self.train_data.shape[0], 2),
        ]
        error_msgs = [
            "the round up log2 of the number of points, i.e. ceil(log2(no. points))."
            "the points' dimensionality.",
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
            unclean_wires=self.unclean_wires,
            additional_wires=self.train_wires,
            additional_bits=self.train_data,
        )

    def get_quantum_circuit(self, x: np.ndarray):
        """
        Given a binary test vector of size m. This function returns a quantum circuit that does the following:
        1. Load data into the trainings register with a quantum associative memory
        2. Invert the i'th qubit in the trainings register, if the i'th entry of the test vector is 0
        => The sum of the trainings register is equal to m minus the hamming distance to the test vector.
        3. For each trainings qubit equal to |1>, an oracle qubit is rotated by pi/m further towards |1>
        4. Uncompute step 2
        5. Return measurements of the trainings register and the oracle qubit
        => The probability of the oracle qubit being |0> is proportional to the hamming distance
        """
        check_binary(
            x, "All the data needs to be binary, when dealing with the hamming distance"
        )
        rot_angle = np.pi / len(x)

        def quantum_circuit():
            self.qam.circuit()
            for x_, train_wire in zip(x, self.train_wires):
                if x_ == 0:
                    qml.PauliX((train_wire,))
            for train_wire in self.train_wires:
                # QAM ancilla wires are 0 after QAM -> use one of those wires
                qml.CRX(rot_angle, wires=(train_wire, self.qam_ancilla_wires[0]))

            return qml.sample(wires=self.idx_wires + [self.qam_ancilla_wires[0]])

        return quantum_circuit

    def calculate_distances(self, x: np.ndarray) -> List[float]:
        samples = qml.QNode(self.get_quantum_circuit(x), self.backend)().tolist()
        num_zero_ancilla = [0] * len(self.train_data)
        total_ancilla = [0] * len(self.train_data)
        # Count how often a certain point was measured (total_ancilla)
        # and how often the ancilla qubit was zero (num_zero_ancilla)
        for sample in samples:
            idx = bitlist_to_int(sample[:-1])
            if idx < len(self.train_data):
                total_ancilla[idx] += 1
                if sample[-1] == 0:
                    num_zero_ancilla[idx] += 1
        # Get prob for ancilla qubit to be equal to 0
        for i, (total_anc_, num_zero_anc) in enumerate(
            zip(total_ancilla, num_zero_ancilla)
        ):
            # 0 <= num_zero_ancilla[i] / total_ancilla[i] <= 1. Hence if total_ancilla[i] == 0, we have no information
            # about the distance, but due to the rot_angle it can't be greater than 1, therefore we set it to 1
            num_zero_ancilla[i] = 1 if total_anc_ == 0 else num_zero_anc / total_anc_
        return num_zero_ancilla

    @staticmethod
    def get_necessary_wires(train_data: np.ndarray) -> Tuple[int, int, int]:
        return (
            int(np.ceil(np.log2(train_data.shape[0]))),
            len(train_data[0]),
            max(len(train_data[0]), 2),
        )

    def get_representative_circuit(self, X: np.ndarray) -> str:
        circuit = qml.QNode(self.get_quantum_circuit(X[0]), self.backend)
        circuit.construct([], {})
        return circuit.qtape.to_openqasm()

    def heatmap_meaningful(self) -> bool:
        return False


class SimpleFidelityQkNN(SimpleQkNN):
    def __init__(
        self,
        train_data,
        train_labels,
        k: int,
        train_wires: List[int],
        test_wires: List[int],
        idx_wires: List[int],
        swap_wires: List[int],
        ancilla_wires: List[int],
        backend: qml.Device,
        unclean_wires=None,
    ):
        super(SimpleFidelityQkNN, self).__init__(train_data, train_labels, k, backend)

        self.prepped_points = self.prep_data(self.train_data)
        self.prepped_points = self.repeat_data_til_next_power_of_two(self.prepped_points)

        self.unclean_wires = [] if unclean_wires is None else unclean_wires
        self.train_wires = train_wires
        self.test_wires = test_wires
        self.ancilla_wires = ancilla_wires
        self.idx_wires = idx_wires
        self.swap_wires = swap_wires

        wire_types = ["train", "test", "idx", "swap", "ancilla", "unclean"]
        num_wires = [
            ceil_log2(train_data.shape[1] + 1),
            ceil_log2(train_data.shape[1] + 1),
            ceil_log2(train_data.shape[0]),
            1,
            3,
        ]
        error_msgs = [
            "ceil(log2(train_datas' dimensionality + 1)).",
            "ceil(log2(train_datas' dimensionality + 1)).",
            "ceil(log2(size of train_data)).",
            "1.",
            "3.",
        ]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-1], num_wires, error_msgs)

        self.loader = TreeLoader(
            self.prepped_points,
            self.idx_wires,
            self.train_wires,
            self.ancilla_wires,
            unclean_wires=self.swap_wires + self.unclean_wires,
        )

    @staticmethod
    def prep_data(data: np.ndarray):
        """
        This function takes in a list of points and does the following:
        1. Normalizes the list of points.
        2. Adds another dimension to each point. If the point was previously 0 in each dimension, then its additional
        dimension is set to 1. Otherwise, the point's additiojnal dimension is set to 0.
        3. Each point gets added more additional dimensions, until its dimensionality is a power of 2.
        4. Return the newly created points
        """
        # Normalize
        norms = np.linalg.norm(data, axis=1)
        zero_elements = np.where(norms == 0)
        norms[zero_elements] = 1
        data = data / norms[:, None]

        # Add another dimension to avoid 0 vectors
        # new dimension is 1 for 0 vectors and 0 otherwise
        new_column = np.zeros(data.shape[0])
        new_column[zero_elements] = 1
        data = np.append(data, new_column[:, None], axis=1)

        # Ensure the number of dimensions is a power of 2
        next_power = 2 ** ceil_log2(data.shape[1])
        next_power = max(next_power, 1)
        data = np.pad(
            data,
            [(0, 0), (0, next_power - data.shape[1])],
            mode="constant",
            constant_values=0,
        )

        return data

    @staticmethod
    def repeat_data_til_next_power_of_two(data):
        """
        Given a list, this function adds the elements of the list again, until the total amount of elements is
        a power of 2.
        """
        next_power = 2 ** ceil_log2(data.shape[0])
        missing_till_next_power = next_power - data.shape[0]
        return np.vstack((data, data[:missing_till_next_power]))

    def get_quantum_circuit(self, x):
        """
        Given a test point, this function returns a quantum circuit that does a Swap-Test between the trainings points
        and the test point. This works as follows:
        1. Load index register with a Walsh-Hadamard transform
        2. Load the trainings data into the trainings register, depending on the index in the index register
        3. Load the test point into the test register
        4. Execute a Swap-test on the trainings and the test register
        5. Return measurements on index register and the swap-qubit
        => fidelity between a trainings point and the test vector is equal to P(swap-qubit = |0>) - P(swap-qubit = |1>)
        """

        def quantum_circuit():
            # Load in training data
            for i in self.idx_wires:
                qml.Hadamard((i,))
            self.loader.circuit()

            # Load test point x
            TreeLoader(
                x[None, :], self.swap_wires, self.test_wires, self.ancilla_wires
            ).circuit()

            # Swap test
            qml.Hadamard((self.swap_wires[0],))
            for train_wire, test_wire in zip(self.train_wires, self.test_wires):
                qml.CSWAP((self.swap_wires[0], train_wire, test_wire))
            qml.Hadamard((self.swap_wires[0],))

            return qml.sample(wires=self.idx_wires + self.swap_wires)

        return quantum_circuit

    def calculate_distances(self, x) -> np.ndarray:
        samples = qml.QNode(self.get_quantum_circuit(x), self.backend)().tolist()
        num_zero_swap = [0] * len(self.train_data)
        idx_count = [0] * len(self.train_data)
        # Count how often a certain index was measured (idx_count)
        # and how often the swap qubit was zero (num_zero_ancilla)
        for sample in samples:
            # Modulo, since we looped the data, until the number of points is a power of 2
            idx = bitlist_to_int(sample[:-1]) % self.train_data.shape[0]
            idx_count[idx] += 1
            if sample[-1] == 0:
                num_zero_swap[idx] += 1

        distances = np.ones((self.train_data.shape[0],))
        for i, (idx_c_, num_zero_s_) in enumerate(zip(idx_count, num_zero_swap)):
            if idx_c_ != 0:
                zero_prob = num_zero_s_ / idx_c_
                # fidelitiy = zero_prob - one_prob = zero_prob - (1 - zero_prob) = 2*zero_prob - 1
                fidelity = 2 * zero_prob - 1
                # Barres distance is sqrt(2 - 2*sqrt(fidelity)). Therefore it suffices to maximise the fidelity
                # maximising fidelity is equivalent to minimising -1*fidelity
                distances[i] = -1 * fidelity

        return distances

    # Override so that we can use 'prep_data'
    def label_points(self, X) -> List[int]:
        if self.backend is None:
            raise ValueError("The quantum backend may not be None!")
        new_labels = []
        X = self.prep_data(X)
        for x in X:
            new_labels.append(self.label_point(x))
        return new_labels

    @staticmethod
    def get_necessary_wires(train_data):
        if not isinstance(train_data, np.ndarray):
            train_data = np.array(train_data)
        train_data = SimpleFidelityQkNN.repeat_data_til_next_power_of_two(
            SimpleFidelityQkNN.prep_data(train_data)
        )
        return (
            ceil_log2(train_data.shape[1] + 1),
            ceil_log2(train_data.shape[1] + 1),
            ceil_log2(train_data.shape[0]),
            1,
            3,
        )

    def get_representative_circuit(self, X) -> str:
        x = self.prep_data(X[:1])[0]
        circuit = qml.QNode(self.get_quantum_circuit(x), self.backend)
        circuit.construct([], {})
        return circuit.qtape.to_openqasm()

    def heatmap_meaningful(self):
        return True


class SimpleAngleQkNN(SimpleQkNN):
    def __init__(
        self,
        train_data,
        train_labels,
        k: int,
        train_wires: List[int],
        idx_wires: List[int],
        swap_wires: List[int],
        ancilla_wires: List[int],
        backend: qml.Device,
        unclean_wires=None,
    ):
        super(SimpleAngleQkNN, self).__init__(train_data, train_labels, k, backend)

        self.prepped_points = self.prep_data(self.train_data)
        self.prepped_points = self.repeat_data_til_next_power_of_two(self.prepped_points)

        self.unclean_wires = [] if unclean_wires is None else unclean_wires
        self.train_wires = train_wires
        self.ancilla_wires = ancilla_wires
        self.idx_wires = idx_wires
        self.swap_wires = swap_wires

        wire_types = ["train", "idx", "swap", "ancilla", "unclean"]
        num_wires = [
            ceil_log2(train_data.shape[1] + 1),
            ceil_log2(train_data.shape[0]),
            1,
            3,
        ]
        error_msgs = [
            "ceil(log2(train_datas' dimensionality + 1)).",
            "ceil(log2(size of train_data)).",
            "1.",
            "3.",
        ]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-1], num_wires, error_msgs)

        self.loader = TreeLoader(
            self.prepped_points,
            self.idx_wires,
            self.train_wires,
            self.ancilla_wires,
            unclean_wires=self.unclean_wires,
            control_wires=self.swap_wires,
        )

    @staticmethod
    def prep_data(data: np.ndarray):
        """
        This function takes in a list of points and does the following:
        1. Normalizes the list of points.
        2. Adds another dimension to each point. If the point was previously 0 in each dimension, then its additional
        dimension is set to 1. Otherwise, the point's additiojnal dimension is set to 0.
        3. Each point gets added more additional dimensions, until its dimensionality is a power of 2.
        4. Return the newly created points
        """
        # Normalize
        norms = np.linalg.norm(data, axis=1)
        zero_elements = np.where(norms == 0)
        norms[zero_elements] = 1
        data = data / norms[:, None]

        # Add another dimension to avoid 0 vectors
        # new dimension is 1 for 0 vectors and 0 otherwise
        new_column = np.zeros(data.shape[0])
        new_column[zero_elements] = 1
        data = np.append(data, new_column[:, None], axis=1)

        # Ensure the number of dimensions is a power of 2
        next_power = 2 ** ceil_log2(data.shape[1])
        next_power = max(next_power, 1)
        data = np.pad(
            data,
            [(0, 0), (0, next_power - data.shape[1])],
            mode="constant",
            constant_values=0,
        )

        return data

    @staticmethod
    def repeat_data_til_next_power_of_two(data):
        """
        Given a list, this function adds the elements of the list again, until the total amount of elements is
        a power of 2.
        """
        next_power = 2 ** ceil_log2(data.shape[0])
        missing_till_next_power = next_power - data.shape[0]
        data = np.vstack((data, data[:missing_till_next_power]))
        return data

    def get_quantum_circuit(self, x):
        """
        Given a test point, this function returns a quantum circuit that does a variant of a Swap-Test between the
        trainings points and the test point. This works as follows:
        1. Use a Hadamard operation on the swap qubit
        2. Load index register with a Walsh-Hadamard transform
        3. If the swap-qubit is equal to |1>, load the trainings data into the trainings register, depending on the
        index in the index register
        3. If the swap-qubit is equal to |0>, load the test point into the test register
        4. Use a Hadamard operation on the swap qubit
        5. Return measurements on index register and the swap-qubit
        => The angle between a trainings point and the test point can be calculated by the probability of the swap-qubit
        to be zero. P(swap-qubit = 0) = (1 + <x|y>)/2 and P(swap-qubit = 1) = (1 - <x|y>)/2
        """

        def quantum_circuit():
            # Init swap wire
            qml.Hadamard((self.swap_wires[0],))

            # Load in training data
            for i in self.idx_wires:
                qml.Hadamard((i,))
            self.loader.circuit()

            # Load test point x
            qml.PauliX((self.swap_wires[0],))
            TreeLoader(
                x[None, :],
                None,
                self.train_wires,
                self.ancilla_wires,
                control_wires=self.swap_wires,
            ).circuit()

            qml.Hadamard((self.swap_wires[0],))

            return qml.sample(wires=self.idx_wires + self.swap_wires)

        return quantum_circuit

    def calculate_distances(self, x) -> np.ndarray:
        samples = qml.QNode(self.get_quantum_circuit(x), self.backend)().tolist()
        num_one_swap = [0] * len(self.train_data)
        idx_count = [0] * len(self.train_data)
        # Count how often a certain index was measured (idx_count)
        # and how often the swap qubit was zero (num_zero_ancilla)
        for sample in samples:
            # Modulo, since we looped the data, until the number of points is a power of 2
            idx = bitlist_to_int(sample[:-1]) % self.train_data.shape[0]
            idx_count[idx] += 1
            if sample[-1] == 1:
                num_one_swap[idx] += 1

        distances = np.ones((self.train_data.shape[0],))
        for i, (idx_c_, num_one_s_) in enumerate(zip(idx_count, num_one_swap)):
            if idx_c_ != 0:
                # The swap test variant used, results in zero_prob = (1 + <Psi|Phi>)/2 and one_prob = (1 - <Psi|Phi>)/2,
                # where |Psi> and |Phi> are the quantum states that get compared
                one_prob = num_one_s_ / idx_c_
                # angle = arccos(<Psi | Phi>) and we want to minimize the angle
                # We want to minimize the angle between the different states. Thus, if we maximize <Psi|Phi>,
                # we also minimize the angle. Further, maximizing zero_prob = (1 + <Psi|Phi)/2 is equivalent.
                # Maximizing zero_prob is the same as minimizing one_prob
                distances[i] = one_prob

        return distances

    # Override so that we can use 'prep_data'
    def label_points(self, X) -> List[int]:
        if self.backend is None:
            raise ValueError("The quantum backend may not be None!")
        new_labels = []
        X = self.prep_data(X)
        for x in X:
            new_labels.append(self.label_point(x))
        return new_labels

    @staticmethod
    def get_necessary_wires(train_data):
        if not isinstance(train_data, np.ndarray):
            train_data = np.array(train_data)
        train_data = SimpleFidelityQkNN.repeat_data_til_next_power_of_two(
            SimpleFidelityQkNN.prep_data(train_data)
        )
        return (
            ceil_log2(train_data.shape[1] + 1),
            ceil_log2(train_data.shape[0]),
            1,
            3,
        )

    def get_representative_circuit(self, X) -> str:
        x = self.prep_data(X[:1])[0]
        circuit = qml.QNode(self.get_quantum_circuit(x), self.backend)
        circuit.construct([], {})
        return circuit.qtape.to_openqasm()

    def heatmap_meaningful(self):
        return True
