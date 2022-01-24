from pyquil import Program
from pyquil.quilatom import MemoryReference
from pyquil.gates import RY, RZ, CNOT


def create_circuit(q_num: int) -> Program:
    """
    Implements circuit A from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
    of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

    :param q_num:
    :return: function that constructs the circuit
    """
    gates_num = int((q_num * q_num - q_num) / 2)
    params_per_gate = 15

    p = Program()
    params = p.declare("params", "REAL", gates_num * params_per_gate)
    param_offset = 0

    for i in range(q_num - 1):
        for j in range(q_num - 1 - i):
            _add_two_qubit_gate(p, j, j + i + 1, params, param_offset)
            param_offset += params_per_gate

    return p


def _add_single_qubit_gate(p: Program, qubit: int, params: MemoryReference, offset: int):
    p += RZ(params[offset], qubit)
    p += RY(params[offset + 1], qubit)
    p += RZ(params[offset + 2], qubit)


def _add_two_qubit_gate(
    p: Program, qubit1: int, qubit2: int, params: MemoryReference, offset: int
):
    _add_single_qubit_gate(p, qubit1, params, offset)
    _add_single_qubit_gate(p, qubit2, params, offset + 3)

    p += CNOT(qubit2, qubit1)

    p += RZ(params[offset + 6], qubit1)
    p += RY(params[offset + 7], qubit2)

    p += CNOT(qubit1, qubit2)
    p += RY(params[offset + 8], qubit2)
    p += CNOT(qubit2, qubit1)

    _add_single_qubit_gate(p, qubit1, params, offset + 9)
    _add_single_qubit_gate(p, qubit2, params, offset + 12)


if __name__ == "__main__":
    test = create_circuit(4)
    print(test)
