import pennylane as qml
import numpy as np
from typing import List
from ..ccnot import adaptive_ccnot
from ..controlled_unitaries import get_controlled_one_qubit_unitary
from ..utils import check_if_values_are_binary
from ..check_wires import check_wires_uniqueness, check_num_wires


class QAM:
    def __init__(self, X, register_wires, ancilla_wires, unclean_wires=None,
                 amplitudes=None, additional_bits=None, additional_wires=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        self.X = X
        if not check_if_values_are_binary(self.X):
            raise ValueError("A QAM (Quantum Associative Memory) can only load binary data")
        self.xor_X = self.create_xor_X(X)

        if additional_bits is not None:
            if not check_if_values_are_binary(additional_bits):
                raise ValueError("A QAM (Quantum Associative Memory) can only load binary additional bits")
            if not isinstance(additional_bits, np.ndarray):
                additional_bits = np.array(additional_bits)
            self.xor_additional_bits = self.create_xor_X(additional_bits)
        else:
            self.xor_additional_bits = additional_bits

        self.register_wires = list(register_wires)
        self.ancilla_wires = list(ancilla_wires)
        self.additional_wires = [] if additional_wires is None else additional_wires
        self.unclean_wires = [] if unclean_wires is None else unclean_wires  # unclean wires are like ancilla wires, but they are not guaranteed to be 0

        ancillas_needed = min(X.shape[1], 3)
        wire_types = ["register", "ancilla", "additional", "unclean"]
        num_wires = [X.shape[1], ancillas_needed]
        error_msgs = ["the points' dimensionality.", str(ancillas_needed)+"."]
        check_wires_uniqueness(self, wire_types)
        check_num_wires(self, wire_types[:-2], num_wires, error_msgs)

        if additional_bits is not None:
            if additional_wires is None or len(additional_wires) < additional_bits.shape[1]:
                raise qml.QuantumFunctionError(
                    "The number of additional wires must be at least the same as the dimension of the additional bits."
                )

        self.load_wire = self.ancilla_wires[0]
        self.control_wire = self.ancilla_wires[1]
        self.ancilla_wires = self.ancilla_wires[2:]


        if amplitudes is None:
            self.amplitudes = [1/np.sqrt(X.shape[0], dtype=np.float64)]*X.shape[0]
        else:
            self.amplitudes = amplitudes


        self.rotation_circuits = self.prepare_rotation_circuits()

    def create_xor_X(self, X):
        xor_X = np.zeros(X.shape)
        for i in range(X.shape[0]-1, 0, -1):
            xor_X[i] = np.bitwise_xor(X[i], X[i-1])
        xor_X[0] = X[0]
        return xor_X

    def abs2(self, x):
        return x.real**2 + x.imag**2

    def prepare_rotation_circuits(self) -> List:
        # rotation_matrices = np.zeros((self.xor_X.shape[0], 4), dtype=np.float64)
        rotation_circuits = []
        prev_sum = 1
        rotation_matrix = np.zeros((2, 2), dtype=np.complex128)
        for i in range(self.xor_X.shape[0]-1):
            next_sum = prev_sum - self.abs2(self.amplitudes[i])
            # print(f"prev_sum = {prev_sum}")
            # print(f"next_sum = {next_sum}")
            diag = np.sqrt(next_sum / prev_sum)
            rotation_matrix[0, 0] = diag
            rotation_matrix[1, 1] = diag

            sqrt_prev_sum = np.sqrt(prev_sum)
            rotation_matrix[0, 1] = self.amplitudes[i] / sqrt_prev_sum
            rotation_matrix[1, 0] = -self.amplitudes[i].conjugate() / sqrt_prev_sum
            # rotation_matrices[i] = np.array(compute_zyz_decomposition(rotation_matrix))
            rotation_circuits.append(get_controlled_one_qubit_unitary(rotation_matrix))
            prev_sum = next_sum

        rotation_matrix[0, 0] = 0
        rotation_matrix[1, 1] = 0

        sqrt_prev_sum = np.sqrt(prev_sum)
        rotation_matrix[0, 1] = self.amplitudes[-1] / sqrt_prev_sum
        rotation_matrix[1, 0] = -self.amplitudes[-1].conjugate() / sqrt_prev_sum
        # rotation_matrices[-1] = np.array(compute_zyz_decomposition(rotation_matrix))
        rotation_circuits.append(get_controlled_one_qubit_unitary(rotation_matrix))

        return rotation_circuits

    def flip_control_if_reg_equals_x(self, x):
        # Flip all wires to one, if reg == x
        for i in range(len(x)):
            if x[i] == 0:
                qml.PauliX(self.register_wires[i])
        # Check if all wires are one
        adaptive_ccnot(self.register_wires, self.ancilla_wires, self.unclean_wires, self.control_wire)
        # Uncompute flips
        for i in range(len(x)):
            if x[i] == 0:
                qml.PauliX(self.register_wires[i])

    def load_x(self, x):
        for idx in range(len(x)):
            if x[idx] == 1:
                qml.CNOT((self.load_wire, self.register_wires[idx]))

    def load_additional_bits(self, bits):
        for idx in range(len(bits)):
            if bits[idx] == 1:
                qml.CNOT((self.load_wire, self.additional_wires[idx]))

    def circuit(self):
        qml.PauliX(self.load_wire)
        for i in range(self.xor_X.shape[0]):

            # Load xi
            self.load_x(self.xor_X[i])
            if self.xor_additional_bits is not None:
                self.load_additional_bits(self.xor_additional_bits[i])
            # qml.Snapshot(f"{i}: loaded {self.X[i]}")  # Debug

            # CNOT load to control
            qml.CNOT(wires=(self.load_wire, self.control_wire))
            # qml.Snapshot(f"{i}: set control to load") # Debug

            # Controlled rotation i from control on load
            self.rotation_circuits[i](self.control_wire, self.load_wire)
            # qml.Snapshot(f"{i}: controlled rotation on load") # Debug

            # Flip control wire, if register == xi
            self.flip_control_if_reg_equals_x(self.X[i])
            # qml.Snapshot(f"{i}: controlled flipped, if reg equals {self.X[i]}")   # Debug

            # Since we are using xor_X, we don't need to uncompute the register, if load == 1
            # xor_X[0] = X[0], but xor_X[i] = X[i] xor X[i-1] for all i > 0
            # After iteration i, reg=X[i]. Now loading xor_X[i+1] results in
            # reg = X[i] xor xor_X[i+1] = X[i] xor (X[i+1] xor X[i]) = (X[i] xor X[i]) xor X[i+1] = 0 xor X[i+1] = X[i+1]
            # qml.Snapshot(f"{i}: qam after {i}'th data point") # Debug

    def get_circuit(self):
        return self.circuit

    def inv_circuit(self):
        self.get_inv_circuit()()

    def get_inv_circuit(self):
        return qml.adjoint(self.circuit)
