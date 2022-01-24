from typing import Tuple

import numpy as np
from pyquil import Program, get_qc
from pyquil.gates import RY, RZ, MEASURE
from pyquil.quilatom import MemoryReference


def create_circuit(q_num: int) -> Tuple[Program, int]:
    """
    Implements circuit B from J. Romero, J. P. Olson, and A. Aspuru-Guzik, “Quantum autoencoders for efficient compression
    of quantum data,” Quantum Sci. Technol., vol. 2, no. 4, p. 045001, Dec. 2017, doi: 10.1088/2058-9565/aa8072.

    :param q_num:
    :return: function that constructs the circuit
    """
    gates_num = 2 * q_num + q_num * (q_num - 1)
    params_per_gate = 3

    p = Program()
    params = p.declare("params", "REAL", gates_num * params_per_gate)
    ro = p.declare("ro", "BIT", q_num)
    param_offset = 0

    for i in range(q_num):
        _add_single_qubit_gate(p, i, params, param_offset)
        param_offset += params_per_gate

    for i in range(q_num):
        for j in range(q_num):
            if i != j:
                _add_controlled_one_qubit_gate(p, i, j, params, param_offset)
                param_offset += params_per_gate

    for i in range(q_num):
        _add_single_qubit_gate(p, i, params, param_offset)
        p += MEASURE(i, ro[i])
        param_offset += params_per_gate

    return p, gates_num * params_per_gate


def _add_single_qubit_gate(p: Program, qubit: int, params: MemoryReference, offset: int):
    p += RZ(params[offset], qubit)
    p += RY(params[offset + 1], qubit)
    p += RZ(params[offset + 2], qubit)


def _add_controlled_one_qubit_gate(
    p: Program,
    control_qubit: int,
    target_qubit: int,
    params: MemoryReference,
    offset: int,
):
    p += RZ(params[offset], target_qubit).controlled(control_qubit)
    p += RY(params[offset + 1], target_qubit).controlled(control_qubit)
    p += RZ(params[offset + 2], target_qubit).controlled(control_qubit)


if __name__ == "__main__":
    p, params_num = create_circuit(4)
    p.wrap_in_numshots_loop(1000)

    qc = get_qc("4q-qvm")
    executable = qc.compile(p)
    executable.write_memory(
        region_name="params", value=np.random.random(params_num) * 2 * np.pi
    )
    bit_strings = qc.run(executable).readout_data.get("ro")
    probabilities = np.mean(bit_strings, 0)

    print(probabilities)
