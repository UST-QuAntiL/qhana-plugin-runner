from typing import Tuple

import numpy as np
from pyquil import Program, get_qc
from pyquil.gates import RY, CNOT, MEASURE, RX, RZ
from pyquil.quilatom import MemoryReference


def create_circuit(q_num: int, layers_num: int) -> Tuple[Program, int]:
    """
    Implements the hardware efficient ansatz from A. Kandala et al., “Hardware-efficient Variational Quantum
    Eigensolver for Small Molecules and Quantum Magnets,” Nature, vol. 549, no. 7671, pp. 242–246, Sep. 2017,
    doi: 10.1038/nature23879.

    :param q_num: number of qubits
    @param layers_num: number of layers
    :return: function that constructs the circuit
    """
    parametric_gates_num = q_num * (layers_num + 1)
    params_per_gate = 3

    p = Program()
    input_values = p.declare("input", "REAL", q_num)
    params = p.declare("params", "REAL", parametric_gates_num * params_per_gate)
    ro = p.declare("ro", "BIT", q_num)
    param_offset = 0

    # input encoding
    for i in range(q_num):
        p += RX(input_values[i], i)

    for i in range(layers_num):
        # layer of single-qubit rotations
        for i in range(q_num):
            _add_single_qubit_gate(p, i, params, param_offset)
            param_offset += params_per_gate

        # entanglement with two layers of CNOTs
        for i in range(0, q_num // 2):
            p += CNOT(2 * i, 2 * i + 1)

        for i in range(0, (q_num - 1) // 2):
            p += CNOT(2 * i + 1, 2 * i + 2)

    # last layer of single-qubit rotations
    for i in range(q_num):
        _add_single_qubit_gate(p, i, params, param_offset)
        param_offset += params_per_gate
        p += MEASURE(i, ro[i])

    return p, parametric_gates_num * params_per_gate


def _add_single_qubit_gate(p: Program, qubit: int, params: MemoryReference, offset: int):
    p += RZ(params[offset], qubit)
    p += RX(params[offset + 1], qubit)
    p += RZ(params[offset + 2], qubit)


if __name__ == "__main__":
    p, params_num = create_circuit(4, 1)
    p.wrap_in_numshots_loop(1000)

    qc = get_qc("4q-qvm")
    executable = qc.compile(p)
    executable.write_memory(
        region_name="params", value=np.random.random(params_num) * 2 * np.pi
    )
    bit_strings = qc.run(executable).readout_data.get("ro")
    probabilities = np.mean(bit_strings, 0)

    print(probabilities)
