from pyquil import Program
from pyquil.quilatom import MemoryReference
from pyquil.gates import RY, RZ, CNOT


def create_circuit(q_num: int) -> Program:
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
    params = p.declare("params", "REAL", gates_num * params_per_gate)
    param_offset = 0

    for i in range(q_num):
        p += RY(params[param_offset], i)
        param_offset += params_per_gate

    for i in range(1, q_num):
        for j in range(i):
            p += CNOT(j, i)

    for i in range(q_num):
        p += RY(params[param_offset], i)
        param_offset += params_per_gate

    return p


if __name__ == "__main__":
    test = create_circuit(4)
    print(test)
