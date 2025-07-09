from typing import List

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace


def generate_reduced_substatevectors(
    qasm_circuit: str, qubit_intervals: List[List[int]]
) -> List[np.ndarray]:
    """
    Generates reduced substatevectors from a given QASM circuit based on specified qubit intervals.

    This function extracts reduced statevectors corresponding to the specified qubit intervals.
    If no intervals are provided (`None` or an empty list), the **entire quantum state** is used.

    ### Behavior:
    - If `qubit_intervals` is `None` or an empty list (`[]`), the function sets it to `[[0, last_qubit_index]]`,
      meaning the full quantum state is used.
    - Otherwise, it extracts the statevector for each given interval `[a_i, b_i]`.

    ### Constraints:
    - Each interval `[a_i, b_i]` in `qubit_intervals` must satisfy:
      - `0 <= a_i <= b_i`
      - `b_i < a_(i+1)` (i.e., intervals must be disjoint and in ascending order)
    - The highest qubit index in any interval must not exceed the maximum qubit index in the circuit.

    ### Note:
    - The function **does not** validate whether the intervals are disjoint or within range;
      validation should be handled at the user input level.

    ### Parameters:
        qasm_circuit (str): The QASM string representation of a quantum circuit.
        qubit_intervals (List[List[int]]): A list of intervals, where each interval is a list `[start, end]`.
                                          If `None` or empty, the entire statevector is used.

    ### Returns:
        List[np.ndarray]: A list of reduced substatevectors corresponding to the specified intervals.
    """

    circuit = QuantumCircuit.from_qasm_str(qasm_circuit)
    last_qubit_index = circuit.num_qubits - 1

    # If qubit_intervals is None or an empty list, set the default range to the full state
    if qubit_intervals is None or len(qubit_intervals) == 0:
        qubit_intervals = [[0, last_qubit_index]]

    # Validate input intervals
    start, end = qubit_intervals[-1]
    if end > last_qubit_index:
        raise ValueError(
            f"The last Interval ends at {end}, which exceeds the last qubit index ({last_qubit_index})."
        )
    density = DensityMatrix.from_instruction(circuit)
    density_matrices = []

    for i, interval in enumerate(qubit_intervals):
        lower, upper = interval
        density = trace_out_qubits(0, lower - 1, density)

        last_qubit_index -= lower
        for j, tup in enumerate(qubit_intervals):
            qubit_intervals[j][0] -= lower
            qubit_intervals[j][1] -= lower

        density_of_state = trace_out_qubits(upper - lower + 1, last_qubit_index, density)
        density_matrices.append(density_of_state)

    output = []
    for density in density_matrices:
        dmatrix = density
        eigenvals, eigenvecs = np.linalg.eig(dmatrix)
        idx = np.where(np.isclose(eigenvals, 1, atol=1e-6))[0]

        if len(idx) == 0:
            raise Exception(
                "Cannot see interval of qubit as an independent state because it has entanglement with other qubits outside of the interval"
            )

        sub_vec = eigenvecs[:, idx[0]]
        output.append(sub_vec)

    return output


def trace_out_qubits(lower: int, upper: int, density_matrix):
    """
    Traces out qubits in the given range from a density matrix.

    - If `lower <= upper`, removes qubits in `[lower, upper]`.
    - If `lower > upper`, returns original density matrix.
    - Checks that indices are within valid range.

    Parameters:
        lower (int): Start qubit index (inclusive).
        upper (int): End qubit index (inclusive).
        density_matrix (DensityMatrix): The density matrix.

    Returns:
        tuple (DensityMatrix, bool):
            - Reduced density matrix or original if no change.
            - Boolean indicating if any qubits were traced out.
    """
    num_qubits = density_matrix.num_qubits
    if lower > upper:
        return density_matrix

    if lower < 0 or upper >= num_qubits:
        raise ValueError(f"Qubit indices out of range: must be in [0, {num_qubits-1}]")

    qubits_to_trace_out = list(range(lower, upper + 1))
    reduced_density = partial_trace(density_matrix, qubits_to_trace_out)
    return reduced_density
