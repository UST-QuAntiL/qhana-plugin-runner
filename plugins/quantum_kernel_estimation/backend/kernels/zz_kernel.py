from typing import List
import pennylane as qml
from .kernel import Kernel
from celery.utils.log import get_task_logger

TASK_LOGGER = get_task_logger(__name__)


class ZZKernel(Kernel):

    def __init__(self, backend, n_qbits, reps, entanglement_pattern_enum):
        super().__init__(backend, n_qbits, reps, entanglement_pattern_enum)

    def get_qbits_needed(self, X, y) -> int:
        return len(X[0])

    @staticmethod
    def feature_map(x) -> float:
        raise NotImplementedError("Method evaluate is not implemented yet!")

    def execute_circuit(self, X, Y, to_calculate, entanglement_pattern):

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

        def ansatz(x, wires_to_use):
            for involved_wires_idx in entanglement_pattern:
                try:
                    involved_wires = [wires_to_use[i] for i in involved_wires_idx]
                    expZZ([x[i] for i in involved_wires_idx], involved_wires)
                except IndexError:
                    TASK_LOGGER.info(f"Index Error got raised when building the ansatz for x = {x}. Involved features of x are {involved_wires_idx}. x length = {len(x)}")
                    raise IndexError(f"Index Error got raised when building the ansatz for x = {x}. Involved features of x are {involved_wires_idx}. x length = {len(x)}")

        def adj_ansatz(y, wires_to_use):
            for involved_wires_idx in reversed(entanglement_pattern):
                involved_wires = [wires_to_use[i] for i in involved_wires_idx]
                adj_expZZ([y[i] for i in involved_wires_idx], involved_wires)

        def full_Hadamard_layer(wires):
            for wire in wires:
                qml.Hadamard(wires=[wire])

        @qml.qnode(self.backend)
        def circuit():
            # projections_and_wires = []
            wires_to_measure = []
            for entry in to_calculate:
                x = X[entry[0]]
                y = Y[entry[1]]
                wires_to_use = entry[2]
                for i in range(self.reps):
                    full_Hadamard_layer(wires_to_use)
                    ansatz(x, wires_to_use)
                for i in range(self.reps):
                    adj_ansatz(y, wires_to_use)
                    full_Hadamard_layer(wires_to_use)
                # projections_and_wires.append((self.get_projector_to_zero(len(wires_to_use)), wires_to_use))
                # TASK_LOGGER.info(f"wires_to_use = {wires_to_use}")
                wires_to_measure.append(wires_to_use)
            # return [qml.expval(qml.Hermitian(projections_and_wires[i][0], wires=projections_and_wires[i][1])) for i in range(len(projections_and_wires))]
            return [qml.probs(wires=wires_to_measure[i]) for i in range(len(wires_to_measure))]

        # return circuit()
        return [result[0] for result in circuit()]
