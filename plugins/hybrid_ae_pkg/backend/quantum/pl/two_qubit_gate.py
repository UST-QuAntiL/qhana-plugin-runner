import pennylane as qml
import pennylane.numpy as np


def add_two_qubit_gate(q1, q2, weights: np.tensor):
    """
    Adds a general two-qubit gate.
    Implements a general two-qubit gate as seen in F. Vatan and C. Williams, “Optimal Quantum Circuits for General
    Two-Qubit Gates,” Phys. Rev. A, vol. 69, no. 3, p. 032315, Mar. 2004, doi: 10.1103/PhysRevA.69.032315.

    :param q1: first input qubit for the gate
    :param q2: second input qubit for the gate
    :param weights: parameters of the gate
    """

    qml.U3(weights[0], weights[1], weights[2], wires=q1)
    qml.U3(weights[3], weights[4], weights[5], wires=q2)

    qml.CNOT(wires=[q2, q1])

    qml.RZ(weights[6], wires=q1)
    qml.RY(weights[7], wires=q2)

    qml.CNOT(wires=[q1, q2])
    qml.RY(weights[8], wires=q2)
    qml.CNOT(wires=[q2, q1])

    qml.U3(weights[9], weights[10], weights[11], wires=q1)
    qml.U3(weights[12], weights[13], weights[14], wires=q2)
