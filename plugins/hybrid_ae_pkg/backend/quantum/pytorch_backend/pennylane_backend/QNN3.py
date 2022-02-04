from typing import Tuple, Callable, List

import pennylane as qml
import pennylane.numpy as np


def constructor(q_num: int) -> Tuple[Callable, int]:
    """
    Implements the circuit from A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, “The power of
    quantum neural networks,” arXiv:2011.00027 [quant-ph], Oct. 2020, Accessed: Nov. 08, 2020. [Online]. Available:
    http://arxiv.org/abs/2011.00027.

    :param q_num: number of qubits
    :return: function that constructs the circuit
    """

    def circuit(inputs: np.tensor, weights: np.tensor) -> List[np.tensor]:
        # angle encoding of the input
        for i in range(q_num):
            qml.RX(inputs[i], wires=i)

        # RY layer
        for i in range(q_num):
            qml.RY(weights[i], wires=i)

        # CNOT layer
        for i in range(1, q_num):
            for j in range(0, i):
                qml.CNOT(wires=[j, i])

        # RY layer
        for i in range(q_num):
            qml.RY(weights[i + q_num], wires=i)

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(q_num)]

    return circuit, 2 * q_num
