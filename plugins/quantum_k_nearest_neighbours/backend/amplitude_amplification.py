import numpy as np
import pennylane as qml

"""
The amplitude amplifications algorithms implemented here, can be found in [0]
[0] Brassard et al. "Quantum amplitude amplification and estimation." Contemporary Mathematics 305 (2002): 53-74. https://arxiv.org/pdf/quant-ph/0005055
"""

def aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr):
    """
    Does itr many grover iterations
    :param state_circuit: A quantum circuit that loads in the state, e.g. Walsh-Hadamard
    :param inv_state_circuit: The inverse of the state_circuit
    :param zero_circuit: A phase oracle for the zero state
    :param oracle_circuit: A phase oracle for the desired state
    :param itr: The number of iterations
    """
    for _ in range(itr):
        state_circuit()
        zero_circuit()
        inv_state_circuit()
        oracle_circuit()


def amplitude_amplification_unique(num_states, state_circuit, inv_state_circuit, zero_circuit, oracle_circuit):
    """
    Does the necessary amount of grover iterations, given that there is only one good/desired state.
    :param state_circuit: A quantum circuit that loads in the state, e.g. Walsh-Hadamard
    :param inv_state_circuit: The inverse of the state_circuit
    :param zero_circuit: A phase oracle for the zero state
    :param oracle_circuit: A phase oracle for the desired state
    """
    # Compute number of iterations
    itr = int(np.pi/4 * np.sqrt(num_states))
    state_circuit()
    aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)


def amplitude_amplification_t_solutions(num_states, state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, t):
    """
    Does the necessary amount of grover iterations, given the number of good results in our quantum state
    :param num_states: The number of states that will be loaded into a superposition by the quantum circuit state_circuit
    :param state_circuit: A quantum circuit that loads in the state, e.g. Walsh-Hadamard
    :param inv_state_circuit: The inverse of the state_circuit
    :param zero_circuit: A phase oracle for the zero state
    :param oracle_circuit: A phase oracle for the desired state
    :param t: The number of good states that will be loaded into a superposition by the quantum circuit state_circuit
    """
    itr = int(np.pi/4 * np.sqrt(num_states/t))
    state_circuit()
    aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)


def get_good_result(result, good_wire=0):
    """
    Checks the results of a quantum circuit for a good/desired result and returns it.
    If there is no good result, then None is returned.
    :param result: A list of samples from a quantum circuit
    :good_wire: The qubit that indicates, if a sample is good or not
    """
    for sample in result:
        if sample[good_wire] == 1:
            return sample[1:]
    return None


def get_exp_search_aa_representative_circuit(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, device,
                                          check_if_good, check_if_good_wire, measure_wires):
    """
    Returns a circuit for the exponential search with amplitude amplification. It only does 2 grover iterations.
    :param state_circuit: A quantum circuit that loads in the state, e.g. Walsh-Hadamard
    :param inv_state_circuit: The inverse of the state_circuit
    :param zero_circuit: A phase oracle for the zero state
    :param oracle_circuit: A phase oracle for the desired state
    :param device: The quantum backend
    :param check_if_good: A circuit that sets a qubit to one, if the state is good/desired
    :param check_if_good_wire: The qubit that will be set to one by the check_if_good circuit
    :param measure_wires: The qubits whose result is desired
    """
    if isinstance(check_if_good_wire, int):
        check_if_good_wire = [check_if_good_wire]

    @qml.qnode(device)
    def circuit():
        state_circuit()
        aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, 2)
        check_if_good()
        return qml.sample(wires=check_if_good_wire + measure_wires)
    return circuit

def exp_searching_amplitude_amplification(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, device,
                                          check_if_good, check_if_good_wire, measure_wires, exp_itr=10):
    """
    Does exponential search with amplitude amplification. An exponential iteration works as follows:
    1. M_float = M_float * c
    2. M = ceil(M_float)
    3. itr = random int between [1, M]
    4. Do grover search with itr many iterations
    5. Check result of grover search. Return if good, do another exponential iteration, if bad
    The algorithm does at most exp_itr many exponential iterations and starts out with M = 1 and c = 1.5
    :param state_circuit: A quantum circuit that loads in the state, e.g. Walsh-Hadamard
    :param inv_state_circuit: The inverse of the state_circuit
    :param zero_circuit: A phase oracle for the zero state
    :param oracle_circuit: A phase oracle for the desired state
    :param device: The quantum backend
    :param check_if_good: A circuit that sets a qubit to one, if the state is good/desired
    :param check_if_good_wire: The qubit that will be set to one by the check_if_good circuit
    :param measure_wires: The qubits whose result is desired
    :param exp_itr: The max number of iterations the exponential search might take.
    """
    if isinstance(check_if_good_wire, int):
        check_if_good_wire = [check_if_good_wire]

    c = 1.5     # 1 < c < 2
    M_float = 1.

    # Check for a good state, without grover
    @qml.qnode(device)
    def circuit():
        state_circuit()
        check_if_good()
        return qml.sample(wires=check_if_good_wire+measure_wires)

    # If we have a good result, return
    # Else continue
    result = get_good_result(circuit())
    if result is not None:
        return result

    # Do exponential iterations, with grover
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

        # If we have a good result, return
        # Else continue
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
