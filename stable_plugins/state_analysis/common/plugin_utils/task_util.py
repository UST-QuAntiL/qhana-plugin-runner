import copy
from typing import Dict, List, Tuple, Union

import numpy as np
from common.state_cropper import generate_reduced_substatevectors
from qiskit import QuantumCircuit

from qhana_plugin_runner.requests import open_url


def validate_qubit_intervals(intervals):
    """
    Validates a list of qubit intervals based on the following conditions:
    - Each interval [start, end] must satisfy start <= end.
    - The first interval must start at 0 (i.e., 0 <= start_0).
    - Consecutive intervals must satisfy end_i < start_{i+1}.
    """
    for i, (start, end) in enumerate(intervals):
        if start > end:
            raise ValueError(f"Invalid interval [{start}, {end}]: start must be <= end.")
        if i == 0 and start < 0:
            raise ValueError(
                f"Invalid first interval [{start}, {end}]: start must be >= 0."
            )
        if i > 0 and intervals[i - 1][1] >= start:
            raise ValueError(
                f"Invalid intervals [{intervals[i-1]}, {start, end}]: "
                "Each end_i must be < start_{i+1}."
            )


def extract_statevectors_from_qasm(qasm_input_list):
    """
    Extracts and processes quantum statevectors from QASM code using given qubit intervals.
    """
    statevectors = []

    for circuit_data in qasm_input_list:
        if isinstance(circuit_data, (list, tuple)) and len(circuit_data) == 2:
            # Case where we have a tuple containing both QASM URL and qubit intervals
            circuit_url, qubit_intervals = circuit_data
        elif isinstance(circuit_data, (list, tuple)) and len(circuit_data) == 1:
            # Case where we have a list/tuple with only the QASM URL
            circuit_url = circuit_data[0]
            qubit_intervals = None
        elif isinstance(circuit_data, str):
            # Case where circuit_data is just a single string (QASM URL)
            circuit_url = circuit_data
            qubit_intervals = None
        else:
            # Raise an error if the input format does not match the expected structures
            raise ValueError(
                f"Each element in 'qasm_input_list' must be either a QASM URL (str), "
                f"a tuple (QASM URL, qubit intervals), or a list/tuple containing a single QASM URL. "
                f"Found: {circuit_data}"
            )
        if qubit_intervals:
            validate_qubit_intervals(qubit_intervals)
        try:
            with open_url(circuit_url) as response:
                qasm_code = response.text
            statevectors.extend(
                generate_reduced_substatevectors(qasm_code, qubit_intervals)
            )
        except Exception as e:
            raise RuntimeError(f"Error retrieving state vectors from QASM code: {e}")
    return statevectors


def convert_nested_list_to_numpy(vectors):
    """
    Converts a nested list of real-imaginary number pairs into a NumPy array of complex numbers.
    """
    try:
        return [np.array([complex(re, im) for (re, im) in vector]) for vector in vectors]
    except Exception as e:
        raise RuntimeError(f"Error converting nested list to NumPy array: {e}")


def generate_numpy_vectors(qasm_inputs=None, vector_data=None):
    """
    Generates NumPy arrays from either QASM inputs or manually provided vector data.
    """
    if qasm_inputs and vector_data:
        raise ValueError("Cannot use both QASM inputs and vector data simultaneously.")
    if not qasm_inputs and not vector_data:
        raise ValueError("Either QASM inputs or vector data must be provided.")

    if qasm_inputs:
        return extract_statevectors_from_qasm(qasm_inputs)
    if vector_data:
        return convert_nested_list_to_numpy(vector_data)

    raise ValueError("Either QASM inputs or vector data must be provided.")


# Markierung für  Schaltkreis zusammengefügen
def merge_circuits(
    qasm_circuit1: str,
    qasm_circuit2: str,
    qubit_interval1: List[List[int]],
    qubit_interval2: List[List[int]],
) -> Tuple[QuantumCircuit, List[List[int]]]:
    """
    Merges two quantum circuits by shifting the qubit indices of the second circuit
    to prevent overlap.

    Args:
        qasm_circuit1 (str): QASM code for the first circuit.
        qasm_circuit2 (str): QASM code for the second circuit.
        qubit_interval1 (List[List[int]]): Qubit intervals for the first circuit.
        qubit_interval2 (List[List[int]]): Qubit intervals for the second circuit.

    Returns:
        Tuple[QuantumCircuit, List[List[int]]]:
            - Merged quantum circuit.
            - Updated qubit intervals.
    """
    qc1 = QuantumCircuit.from_qasm_str(qasm_circuit1)
    qc2 = QuantumCircuit.from_qasm_str(qasm_circuit2)

    num_qc1 = qc1.num_qubits
    num_qc2 = qc2.num_qubits

    # If no interval is provided, the whole state will be used
    if qubit_interval1 is None:
        qubit_interval1 = [[0, num_qc1 - 1]]

    if qubit_interval2 is None:
        qubit_interval2 = [[0, num_qc2 - 1]]

    # Create a new circuit with sufficient qubits
    merged_circuit = QuantumCircuit(num_qc1 + num_qc2)

    # Add qc1 to the merged circuit
    merged_circuit.compose(qc1, qubits=range(num_qc1), inplace=True)

    # Add qc2 with shifted qubit indices
    merged_circuit.compose(qc2, qubits=range(num_qc1, num_qc1 + num_qc2), inplace=True)

    # Shift qubit intervals accordingly
    new_qubit_intervals = qubit_interval1
    for interval in qubit_interval2:
        new_qubit_intervals.append([interval[0] + num_qc1, interval[1] + num_qc1])

    return merged_circuit, new_qubit_intervals


def generating_one_circuit_with_each_of_two_states_one_time(
    qasm_input_list,
) -> Tuple[QuantumCircuit, List[List[int]]]:

    qasm_code_and_intervals = []

    for circuit_data in qasm_input_list:
        if isinstance(circuit_data, (list, tuple)) and len(circuit_data) == 2:
            # Case where we have a tuple containing both QASM URL and qubit intervals
            circuit_url, qubit_intervals = circuit_data
        elif isinstance(circuit_data, (list, tuple)) and len(circuit_data) == 1:
            # Case where we have a list/tuple with only the QASM URL
            circuit_url = circuit_data[0]
            qubit_intervals = None
        elif isinstance(circuit_data, str):
            # Case where circuit_data is just a single string (QASM URL)
            circuit_url = circuit_data
            qubit_intervals = None
        else:
            # Raise an error if the input format does not match the expected structures
            raise ValueError(
                f"Each element in 'qasm_input_list' must be either a QASM URL (str), "
                f"a tuple (QASM URL, qubit intervals), or a list/tuple containing a single QASM URL. "
                f"Found: {circuit_data}"
            )
        if qubit_intervals:
            validate_qubit_intervals(qubit_intervals)

        with open_url(circuit_url) as response:
            qasm_code = response.text
        qasm_code_and_intervals.append((qasm_code, qubit_intervals))

    if len(qasm_code_and_intervals) == 1:
        qasm_code, qubit_intervals = qasm_code_and_intervals[0]
        if qubit_intervals is not None and len(qubit_intervals) == 2:
            try:
                return QuantumCircuit.from_qasm_str(qasm_code), qubit_intervals
            except Exception as e:
                raise Exception(f"Error while parsing: {qasm_code}: {e}")
        else:
            raise ValueError(
                "Invalid input: A single circuit was provided, but the qubit intervals did not define two states."
            )

    if len(qasm_code_and_intervals) == 2:
        qasm_code1, qubit_interval1 = qasm_code_and_intervals[0]
        qasm_code2, qubit_interval2 = qasm_code_and_intervals[1]

        if (qubit_interval1 is not None and len(qubit_interval1) != 1) or (
            qubit_interval2 is not None and len(qubit_interval2) != 1
        ):
            raise ValueError(
                "If two QASM codes are provided, each must define exactly one state if a state is specified. "
                "If not provided, the whole state will be used."
            )

        return merge_circuits(qasm_code1, qasm_code2, qubit_interval1, qubit_interval2)

    raise ValueError(
        f"Invalid input: Expected at most 2 circuits but received {len(qasm_code_and_intervals)}."
    )


def generate_one_circuit_with_two_states(
    qasm_input_list,
) -> Tuple[QuantumCircuit, List[List[int]]]:
    return generating_one_circuit_with_each_of_two_states_one_time(qasm_input_list)


def generate_one_circuit_with_multiple_states_weak(
    qasm_input_list, COPIES_PER_STATE: int
) -> Tuple[QuantumCircuit, List[List[int]]]:

    if not isinstance(COPIES_PER_STATE, int):
        raise TypeError("COPIES_PER_STATE must be an integer.")
    if COPIES_PER_STATE <= 0:
        raise ValueError("COPIES_PER_STATE must be a positive integer greater than 0.")

    base_circuit, base_intervals = (
        generating_one_circuit_with_each_of_two_states_one_time(qasm_input_list)
    )

    # unabhängige Arbeitskopien
    collecting_circuit = copy.deepcopy(base_circuit)
    collecting_interval = copy.deepcopy(base_intervals)

    # statische Referenzen (“Vorlage”)
    add_on_circuit = copy.deepcopy(base_circuit)
    add_on_interval = copy.deepcopy(base_intervals)

    for _ in range(1, COPIES_PER_STATE):
        collecting_circuit, collecting_interval = merge_circuits(
            collecting_circuit.qasm(),
            add_on_circuit.qasm(),
            collecting_interval,
            add_on_interval,
        )

    return collecting_circuit, collecting_interval
