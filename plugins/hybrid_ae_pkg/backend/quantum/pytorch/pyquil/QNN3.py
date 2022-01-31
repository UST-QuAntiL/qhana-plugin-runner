from typing import Tuple

import numpy as np
from pyquil import Program, get_qc
from pyquil.gates import RY, CNOT, MEASURE, RX


def create_circuit(q_num: int) -> Tuple[Program, int]:
    """
    Implements the circuit from figure 5 from A. Abbas, D. Sutter, C. Zoufal, A. Lucchi, A. Figalli, and S. Woerner, “The power of
    quantum neural networks,” arXiv:2011.00027 [quant-ph], Oct. 2020, Accessed: Nov. 08, 2020. [Online]. Available:
    http://arxiv.org/abs/2011.00027.

    :param q_num: number of qubits
    :return: function that constructs the circuit
    """
    gates_num = 2 * q_num
    params_per_gate = 1

    p = Program()
    input_values = p.declare("input", "REAL", q_num)
    params = p.declare("params", "REAL", gates_num * params_per_gate)
    ro = p.declare("ro", "BIT", q_num)
    param_offset = 0

    for i in range(q_num):
        p += RX(input_values[i], i)

    for i in range(q_num):
        p += RY(params[param_offset], i)
        param_offset += params_per_gate

    for i in range(1, q_num):
        for j in range(i):
            p += CNOT(j, i)

    for i in range(q_num):
        p += RY(params[param_offset], i)
        p += MEASURE(i, ro[i])
        param_offset += params_per_gate

    return p, gates_num * params_per_gate


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
