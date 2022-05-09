from typing import List
import pennylane as qml
from .kernel import Kernel
from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)


class ZZKernel(Kernel):

    def __init__(self, backend, n_qbits, reps, entanglement_pattern):
        super().__init__(backend, n_qbits, reps, entanglement_pattern)

    @staticmethod
    def feature_map(x) -> float:
        raise NotImplementedError("Method evaluate is not implemented yet!")

    def get_circuit(self):

        def CNOT_chain(wires: List[int]):
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

        def adj_CNOT_chain(wires: List[int]):
            for i in range(len(wires) - 2, -1, -1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])

        def expZZ(x, wires: List[int]):
            CNOT_chain(wires)
            qml.RZ(2. * self.feature_map(x), wires=[wires[-1]])
            adj_CNOT_chain(wires)

        def adj_expZZ(y, wires: List[int]):
            CNOT_chain(wires)
            qml.RZ(-2. * self.feature_map(y), wires=[wires[-1]])
            adj_CNOT_chain(wires)

        def ansatz(x):
            for involved_qbits in self.entanglement_pattern:
                expZZ([x[i] for i in involved_qbits], involved_qbits)

        def adj_ansatz(y):
            for involved_qbits in reversed(self.entanglement_pattern):
                adj_expZZ([y[i] for i in involved_qbits], involved_qbits)

        def full_Hadamard_layer(n_qbits):
            for i in range(n_qbits):
                qml.Hadamard(wires=[i])

        @qml.qnode(self.backend)
        def circuit(x, y):
            for i in range(self.reps):
                full_Hadamard_layer(self.n_qbits)
                ansatz(x)
            for i in range(self.reps):
                adj_ansatz(y)
                full_Hadamard_layer(self.n_qbits)
            projector = self.get_projector_to_zero(self.n_qbits)
            return qml.expval(qml.Hermitian(projector, wires=range(self.n_qbits)))

        return circuit
