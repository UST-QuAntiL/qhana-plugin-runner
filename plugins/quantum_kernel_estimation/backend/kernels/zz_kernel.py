# Copyright 2022 QHAna plugin runner contributors.
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

from typing import List
import pennylane as qml
from .kernel import Kernel
from celery.utils.log import get_task_logger
from abc import abstractmethod, ABCMeta


TASK_LOGGER = get_task_logger(__name__)


class ZZKernel(Kernel, metaclass=ABCMeta):
    """
    This class creates the circuit proposed by Havlíček et al., but for any given feature map.
    The name for this class is inspired by the name of qiskit's implementation.
    Havlíček, V., Córcoles, A.D., Temme, K. et al. Supervised learning with quantum-enhanced feature spaces. Nature 567, 209–212 (2019). https://doi.org/10.1038/s41586-019-0980-2
    """

    def __init__(self, backend, n_qbits, reps, entanglement_pattern_enum):
        super().__init__(backend, n_qbits, reps, entanglement_pattern_enum)

    def get_qbits_needed(self, data_x, data_y) -> int:
        return len(data_x[0])

    @abstractmethod
    def feature_map(self, x) -> float:
        raise NotImplementedError("Method evaluate is not implemented yet!")

    def execute_circuit(self, data_x, data_y, to_calculate, entanglement_pattern):
        """
        Executes circuits foreach entry in to_calculate. An entry in to_calculate contains the information of which data point
        in data_x should be evaluated with which data point in data_y and which qubits should be used for the resulting quantum circuit.
        :param data_x: list of data points
        :param data_y: list of data points
        :param to_calculate: list. Each entry contains an index for data_x, an index for data_y and a set of qubits.
        :param entanglement_pattern: Entanglement pattern that should be used.
        """

        def CNOT_chain(wires: List[int]):
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

        def adj_CNOT_chain(wires: List[int]):
            for i in range(len(wires) - 2, -1, -1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

        def expZZ(x, wires: List[int]):
            CNOT_chain(wires)
            qml.RZ(2.0 * self.feature_map(x), wires=[wires[-1]])
            adj_CNOT_chain(wires)

        def adj_expZZ(y, wires: List[int]):
            CNOT_chain(wires)
            qml.RZ(-2.0 * self.feature_map(y), wires=[wires[-1]])
            adj_CNOT_chain(wires)

        def ansatz(x, wires_to_use):
            for involved_wires_idx in entanglement_pattern:
                try:
                    involved_wires = [wires_to_use[i] for i in involved_wires_idx]
                    expZZ([x[i] for i in involved_wires_idx], involved_wires)
                except IndexError:
                    TASK_LOGGER.info(
                        f"Index Error got raised when building the ansatz for x = {x}. Involved features of x are {involved_wires_idx}. x length = {len(x)}"
                    )
                    raise IndexError(
                        f"Index Error got raised when building the ansatz for x = {x}. Involved features of x are {involved_wires_idx}. x length = {len(x)}"
                    )

        def adj_ansatz(y, wires_to_use):
            for involved_wires_idx in reversed(entanglement_pattern):
                involved_wires = [wires_to_use[i] for i in involved_wires_idx]
                adj_expZZ([y[i] for i in involved_wires_idx], involved_wires)

        def full_Hadamard_layer(wires):
            for wire in wires:
                qml.Hadamard(wires=[wire])

        @qml.qnode(self.backend)
        def circuit():
            """
            Creates the circuit for evaluation
            """
            wires_to_measure = []
            for entry in to_calculate:
                x = data_x[entry[0]]
                y = data_y[entry[1]]
                wires_to_use = entry[2]
                for _ in range(self.reps):
                    full_Hadamard_layer(wires_to_use)
                    ansatz(x, wires_to_use)
                for _ in range(self.reps):
                    adj_ansatz(y, wires_to_use)
                    full_Hadamard_layer(wires_to_use)
                wires_to_measure.append(wires_to_use)
            return [
                qml.probs(wires=wires_to_measure[i]) for i in range(len(wires_to_measure))
            ]

        results = circuit()

        # Probs for measured wires to be |0> and open qasm
        return [result[0] for result in results], circuit.tape.to_openqasm()
