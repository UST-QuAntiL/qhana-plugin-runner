# Copyright 2023 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pennylane as qml
from typing import Callable, List, Optional

"""
The amplitude amplifications algorithms implemented here, can be found in [0] and [1]
[0] Brassard et al. "Quantum amplitude amplification and estimation." Contemporary Mathematics 305 (2002): 53-74. https://arxiv.org/pdf/quant-ph/0005055
[1] M. Boyer, G. Brassard, P. Høyer and A. Tapp (1998), Tight Bounds on Quantum Searching. Fortschr. Phys., 46: 493-505. https://doi.org/10.1002/(SICI)1521-3978(199806)46:4/5<493::AID-PROP493>3.0.CO;2-P
"""


def aa_steps(
    state_circuit: Callable[[], None],
    inv_state_circuit: Callable[[], None],
    zero_circuit: Callable[[], None],
    oracle_circuit: Callable[[], None],
    itr: int,
):
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


def amplitude_amplification_unique(
    num_states: int,
    state_circuit: Callable[[], None],
    inv_state_circuit: Callable[[], None],
    zero_circuit: Callable[[], None],
    oracle_circuit: Callable[[], None],
):
    """
    Does the necessary amount of grover iterations, given that there is only one good/desired state.
    :param state_circuit: A quantum circuit that loads in the state, e.g. Walsh-Hadamard
    :param inv_state_circuit: The inverse of the state_circuit
    :param zero_circuit: A phase oracle for the zero state
    :param oracle_circuit: A phase oracle for the desired state
    """
    # Compute number of iterations
    itr = int((np.pi / 4) * np.sqrt(num_states))
    state_circuit()
    aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)


def amplitude_amplification_t_solutions(
    num_states: int,
    state_circuit: Callable[[], None],
    inv_state_circuit: Callable[[], None],
    zero_circuit: Callable[[], None],
    oracle_circuit: Callable[[], None],
    t: int,
):
    """
    Does the necessary amount of grover iterations, given the number of good results in our quantum state
    :param num_states: The number of states that will be loaded into a superposition by the quantum circuit state_circuit
    :param state_circuit: A quantum circuit that loads in the state, e.g. Walsh-Hadamard
    :param inv_state_circuit: The inverse of the state_circuit
    :param zero_circuit: A phase oracle for the zero state
    :param oracle_circuit: A phase oracle for the desired state
    :param t: The number of good states that will be loaded into a superposition by the quantum circuit state_circuit
    """
    itr = int(np.pi / 4 * np.sqrt(num_states / t))
    state_circuit()
    aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)


def get_good_result(result: List[List[int]], good_wire: int = 0) -> Optional[List[int]]:
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


def get_exp_search_aa_representative_circuit(
    state_circuit: Callable[[], None],
    inv_state_circuit: Callable[[], None],
    zero_circuit: Callable[[], None],
    oracle_circuit: Callable[[], None],
    device: qml.Device,
    check_if_good: Callable[[], None],
    check_if_good_wire: int,
    measure_wires: List[int],
) -> Optional[List[int]]:
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


def exp_searching_amplitude_amplification(
    state_circuit: Callable[[], None],
    inv_state_circuit: Callable[[], None],
    zero_circuit: Callable[[], None],
    oracle_circuit: Callable[[], None],
    device: qml.Device,
    check_if_good: Callable[[], None],
    check_if_good_wire: int,
    measure_wires: List[int],
    exp_itr: int = 10,
) -> Optional[List[int]]:
    """
    Does exponential search with amplitude amplification. An exponential iteration works as follows:
    1. M_float = M_float * c
    2. M = ceil(M_float)
    3. itr = random int between [1, M]
    4. Do grover search with itr many iterations
    5. Check result of grover search. Return if good, do another exponential iteration, if bad
    The algorithm does at most exp_itr many exponential iterations and starts out with M_float = 1 and c = 1.5
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

    c = 1.5  # 1 < c < 2
    M_float = 1.0

    # Check for a good state, without grover
    @qml.qnode(device)
    def circuit():
        state_circuit()
        check_if_good()
        return qml.sample(wires=check_if_good_wire + measure_wires)

    # If we have a good result, return
    # Else continue
    result = get_good_result(circuit())
    if result is not None:
        return result

    # Do exponential iterations, with grover
    for _ in range(exp_itr):  # This should actually go to infinity
        M_float *= c
        M = int(np.ceil(M_float))
        itr = np.random.randint(1, M)

        @qml.qnode(device)
        def circuit():
            state_circuit()
            aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)
            check_if_good()
            return qml.sample(wires=check_if_good_wire + measure_wires)

        # If we have a good result, return
        # Else continue
        result = get_good_result(circuit())
        if result is not None:
            return result

    return None


def lambda_amplitude_amplification(
    num_states: int,
    state_circuit: Callable[[], None],
    inv_state_circuit: Callable[[], None],
    zero_circuit: Callable[[], None],
    oracle_circuit: Callable[[], None],
    device: qml.Device,
    check_if_good: Callable[[], None],
    check_if_good_wire: List[int],
    measure_wires: List[int],
    max_itr: int = 10,
) -> Optional[List[int]]:
    """
    This algorithm works similarly as the exponential search algorithm. One Iteration works as follows:
    1. itr = random int between [1, m]
    2. Do grover search with itr many iterations
    3. Check result of grover search. Return if good.
    4. m = min(lam * m, sqrt(num_states)) and do another iteration.
    The algorithm does at most exp_itr many iterations and starts out with m = 1, lam = 8/7.
    :param num_states: The amount of states in the superposition (bad and good).
    :param state_circuit: A quantum circuit that loads in the state, e.g. Walsh-Hadamard
    :param inv_state_circuit: The inverse of the state_circuit
    :param zero_circuit: A phase oracle for the zero state
    :param oracle_circuit: A phase oracle for the desired state
    :param device: The quantum backend
    :param check_if_good: A circuit that sets a qubit to one, if the state is good/desired
    :param check_if_good_wire: The qubit that will be set to one by the check_if_good circuit
    :param measure_wires: The qubits whose result is desired
    :param exp_itr: The max number of iterations.
    """
    m = 1
    lam = 8 / 7  # 1 < lam < 4/3
    sqrt_num_states = np.sqrt(num_states)

    for _ in range(max_itr):  # This should actually go to infinity
        itr = np.random.randint(0, m)

        @qml.qnode(device)
        def circuit():
            state_circuit()
            aa_steps(state_circuit, inv_state_circuit, zero_circuit, oracle_circuit, itr)
            check_if_good()
            return qml.sample(wires=check_if_good_wire + measure_wires)

        result = get_good_result(circuit())
        if result is not None:
            return result
        m = min(lam * m, sqrt_num_states)

    return None
