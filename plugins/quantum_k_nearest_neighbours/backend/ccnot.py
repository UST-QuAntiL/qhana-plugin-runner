import pennylane as qml


def xor_int(x1, x2):
    return 0 if x1 == x2 else 1


def bit_list(num, num_bits):
    bits = [int(el) for el in bin(num)[2:]]
    bits = [0]*(num_bits - len(bits)) + bits
    return bits


def one_ancilla_ccnot(c_qubits, a_qubit, t_qubit):
    """
    This consists of 4 steps
    1. Partition c_qubits into two sets c1 and c2
    2. Compute unclean ccnot(c1, c2, a_qubit)
    3. Compute unclean ccnot(c2+a_qubit, c1, t)
    4. Uncompute unclean ccnot(c1, c2, a_qubit)
    Given n many control qubits, an unclean ccnot needs n-2 ancilla qubits.
    Therefore, |c1| >= |c2| - 1 and |c2| >= |c1| - 2.
    If |c_qubits| is even => |c1| = |c_qubits|/2 and |c2| = |c_qubits|/2
    If |c_qubits| is uneven => |c1| = lower(|c_qubits|/2) and |c2| = ceil(|c_qubits|/2)
    """
    num_c1 = int(len(c_qubits)/2)
    c1 = c_qubits[:num_c1]
    c2 = c_qubits[num_c1:]
    unclean_ccnot(c1, c2, a_qubit)
    unclean_ccnot(c2+[a_qubit], c1, t_qubit)
    unclean_ccnot(c1, c2, a_qubit)


def clean_ccnot(c_qubits, a_qubits, t_qubit):
    if len(c_qubits) == 0:
        qml.PauliX((t_qubit,))
    elif len(c_qubits) == 1:
        qml.CNOT((c_qubits[0], t_qubit))
    elif len(c_qubits) == 2:
        qml.Toffoli(wires=c_qubits + [t_qubit])
    else:
        qml.Toffoli(wires=c_qubits[:2] + [a_qubits[0]])
        for i in range(2, len(c_qubits)-1):
            qml.Toffoli(wires=[c_qubits[i]] + a_qubits[i-2:i])
        qml.Toffoli(wires=[c_qubits[-1], a_qubits[len(c_qubits)-3], t_qubit])
        for i in range(len(c_qubits)-2, 1, -1):
            qml.Toffoli(wires=[c_qubits[i]] + a_qubits[i-2:i])
        qml.Toffoli(wires=c_qubits[:2] + [a_qubits[0]])


def unclean_ccnot(c_qubits, a_qubits, t_qubit):
    """
    This ccnot operation works, even if the ancilla register has non zero values, i.e. it is not clean
    :param c_qubits:
    :param a_qubits:
    :param t_qubit:
    :return:
    """
    if len(c_qubits) == 0:
        qml.PauliX((t_qubit,))
    elif len(c_qubits) == 1:
        qml.CNOT(wires=c_qubits + [t_qubit])
    elif len(c_qubits) == 2:
        qml.Toffoli(wires=c_qubits + [t_qubit])
    else:
        n = len(c_qubits)
        qml.Toffoli(wires=[c_qubits[-1], a_qubits[-1], t_qubit])
        for i in range(-2, -n+1, -1):
            qml.Toffoli(wires=[c_qubits[i], a_qubits[i], a_qubits[i+1]])
        qml.Toffoli(wires=[c_qubits[0], c_qubits[1], a_qubits[-n+2]])
        for i in range(-n+2, -1):
            qml.Toffoli(wires=[c_qubits[i], a_qubits[i], a_qubits[i+1]])
        qml.Toffoli(wires=[c_qubits[-1], a_qubits[-1], t_qubit])

        for i in range(-2, -n + 1, -1):
            qml.Toffoli(wires=[c_qubits[i], a_qubits[i], a_qubits[i + 1]])
        qml.Toffoli(wires=[c_qubits[0], c_qubits[1], a_qubits[-n + 2]])
        for i in range(-n + 2, -1):
            qml.Toffoli(wires=[c_qubits[i], a_qubits[i], a_qubits[i + 1]])


def main():
    c_qubits = list(range(6))
    a_qubits = list(range(len(c_qubits), len(c_qubits)*2))
    t_qubit = len(c_qubits) + len(a_qubits)
    num_wires = len(c_qubits) + len(a_qubits) + 1
    device = qml.device('default.qubit', wires=num_wires)
    device.shots = 1

    @qml.qnode(device)
    def circuit():
        for wire in a_qubits:
            qml.PauliX((wire,))
        for wire in c_qubits:
            qml.PauliZ((wire,))
        qml.PauliY((t_qubit,))
        unclean_ccnot(c_qubits, a_qubits, t_qubit)
        return qml.probs(c_qubits+a_qubits+[t_qubit])

    print(qml.draw(circuit)())


if __name__ == '__main__':
    main()
