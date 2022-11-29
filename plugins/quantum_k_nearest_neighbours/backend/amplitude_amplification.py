import numpy as np
import pennylane as qml


def aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr):
    for _ in range(itr):
        state_circuit()
        zero_circuit()
        inv_state_circuit()
        oracle_circuit()


def amplitude_amplification_unique(num_states, state_circuit, inv_state_circuit, zero_circuit, oracle_circuit):
    # Amplitude amplification with the grover algorithm
    # Compute number of iterations
    itr = int(np.pi/4 * np.sqrt(num_states))
    state_circuit()
    aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)


def amplitude_amplification_t_solutions(num_states, state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, t):
    itr = int(np.pi/4 * np.sqrt(num_states/t))
    state_circuit()
    aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)


def get_good_result(result, good_wire=0):
    for sample in result:
        if sample[good_wire] == 1:
            return sample[1:]
    return None


def get_exp_search_aa_representative_circuit(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, device,
                                          check_if_good, check_if_good_wire, measure_wires):
    @qml.qnode(device)
    def circuit():
        state_circuit()
        aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, 2)
        check_if_good()
        return qml.sample(wires=check_if_good_wire + measure_wires)
    return circuit

def exp_searching_amplitude_amplification(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, device,
                                          check_if_good, check_if_good_wire, measure_wires, exp_itr=10):
    check_if_good_wire = [check_if_good_wire]

    c = 1.5     # 1 < c < 2
    M_float = 1.

    @qml.qnode(device)
    def circuit():
        state_circuit()
        check_if_good()
        return qml.sample(wires=check_if_good_wire+measure_wires)
        # return qml.sample(wires=range(3+3+3))
    # c_result = circuit()
    # print(c_result)
    # result = get_good_result(c_result, )
    result = get_good_result(circuit())
    if result is not None:
        return result

    for _ in range(exp_itr):     # This should actually go to infinity
        M_float *= c
        M = int(np.ceil(M_float))
        itr = np.random.randint(1, M)

        @qml.qnode(device)
        def circuit():
            state_circuit()
            aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)
            check_if_good()
            return qml.sample(wires=check_if_good_wire+measure_wires)

        # print(qml.draw(circuit)())
        result = get_good_result(circuit())
        if result is not None:
            return result

    return None


def lambda_amplitude_amplification(num_states, state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, device,
                                          check_if_good, check_if_good_wire, measure_wires, exp_itr=10):
    m = 1
    lam = 8/7
    sqrt_num_states = np.sqrt(num_states)

    for _ in range(exp_itr):     # This should actually go to infinity
        itr = np.random.randint(0, m)

        @qml.qnode(device)
        def circuit():
            state_circuit()
            aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)
            check_if_good()
            return qml.sample(wires=check_if_good_wire+measure_wires)

        # print(qml.draw(circuit)())
        result = get_good_result(circuit())
        if result is not None:
            return result
        m = min(lam*m, sqrt_num_states)

    return None

