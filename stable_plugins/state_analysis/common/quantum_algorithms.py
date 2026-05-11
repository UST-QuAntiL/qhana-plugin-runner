# Copyright 2026 QHAna plugin runner contributors.
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

import math
from math import ceil, log2
from typing import Dict, List, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import Permutation


def generate_swaptest_circuit(
    states_circuit,
    first_interval_start,
    first_interval_end,
    second_interval_start,
    second_interval_end,
) -> QuantumCircuit:
    """Generates a Cuircuit that appends a swap test to the original circuit.
    The swap test is performed between two defined quantum state intervals:
    first_interval_start to first_interval_end and second_interval_start to second_interval_end.
    """

    # Add an ancilla qubit for the swap test
    ancilla = QuantumRegister(1, "ancilla")
    states_circuit.add_register(ancilla)

    # Add a classical register to store the measurement result
    cr = ClassicalRegister(1, "c")
    states_circuit.add_register(cr)

    # Apply a Hadamard gate to the ancilla qubit to create superposition
    states_circuit.h(ancilla[0])

    # Compute the interval length (assuming both intervals are the same size and inclusive)
    interval_size = first_interval_end - first_interval_start + 1

    # Apply controlled SWAP gates (Fredkin gates) for each qubit pair in the intervals
    for i in range(interval_size):
        states_circuit.cswap(
            ancilla[0],
            states_circuit.qubits[first_interval_start + i],
            states_circuit.qubits[second_interval_start + i],
        )

    # Apply another Hadamard gate to the ancilla qubit
    states_circuit.h(ancilla[0])

    # Measure the ancilla qubit and store the result in the classical register
    states_circuit.measure(ancilla[0], cr[0])

    # Return the modified circuit code
    return states_circuit


# Markierung Interpreter
def interpret_swaptest_results(zeros: int, ones: int, shots: int) -> bool:
    """
    Interprets the results of the swap test.

    Args:
        zeros (int): Number of times the measurement resulted in '0'.
        ones (int): Number of times the measurement resulted in '1'.
        shots (int): Total number of shots (measurements).

    Returns:
        bool: True if states are likely orthogonal, False if they are likely parallel.
    """
    return zeros < (shots / 2 + 1.645 * math.sqrt(shots) / 2)
