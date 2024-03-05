# regex to find total number of classical bits
import re
import time
from typing import Dict, Union, List, Tuple

import cirq
import numpy as np
from cirq import Circuit, Result
from cirq.contrib.qasm_import import circuit_from_qasm


def simulate_circuit(circuit_qasm: str, execution_options: Dict[str, Union[str, int]]):
    circuit_qasm = circuit_qasm.replace("\r\n", "\n")
    circuit = circuit_from_qasm(circuit_qasm)
    # Zero time indicates no measurements (in qasm code)
    star_time_count = 0
    end_time_count = 0
    # Make a copy of the circuit to keep original, unchanged before adding any measurements.
    circuit_copy = circuit.copy()

    number_qubits = len(list(circuit.all_qubits()))
    register_sizes = _parse_classical_registers(circuit_qasm)

    simulator = cirq.Simulator()

    if circuit.has_measurements():
        star_time_count = time.perf_counter_ns()
        result = simulator.run(
            circuit_copy, repetitions=execution_options["shots"]
        )  # simulation (counts)
        end_time_count = time.perf_counter_ns()

        histogram = _parse_to_histogram(circuit_copy, result, register_sizes)
    else:  # If there are no measurements in qasm code
        shots = execution_options["shots"]
        histogram = {"": shots}

    if execution_options.get("statevector"):
        state_vector = simulator.simulate(
            circuit
        ).final_state_vector  # simulation (statevector)
    else:
        state_vector = None

    # Convert the outcomes to binary format
    binary_histogram = {
        format(outcome, f"0{number_qubits}b")
        if outcome and isinstance(outcome, int)
        else outcome: frequency
        for outcome, frequency in histogram.items()
    }

    metadata = {
        # trace ids (specific to IBM qiskit jobs)
        "jobId": None,
        "qobjId": None,
        # QPU/Simulator information
        "qpuType": "simulator",
        "qpuVendor": "Google",
        "qpuName": "Cirq Simulator",
        "qpuVersion": None,
        "seed": None,  # only for simulators
        "shots": execution_options["shots"],
        # Time information
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "timeTakenIdle": 0,  # idle/waiting time
        "timeTakenCounts_nanosecond": end_time_count - star_time_count,
    }

    return metadata, binary_histogram, state_vector


def _parse_classical_registers(qasm_code: str) -> List[Tuple[str, int]]:
    """
    Parses the classical registers in the QASM code.
    :param qasm_code:
    :return: list of tuples of name and size of the classical registers
    """
    no_comment_qasm = re.sub(r"//.*\n?", "", qasm_code)
    matches = re.findall(r"creg ([a-zA-Z0-9_]+)\[(\d+)];", no_comment_qasm)

    if not matches:
        return []

    register_sizes: List[Tuple[str, int]] = []

    for register_name, size in matches:
        register_sizes.append((register_name, int(size)))

    return register_sizes


def _parse_to_histogram(
    circuit: Circuit, result: Result, register_sizes: List[Tuple[str, int]]
) -> Dict[str, int]:
    """
    Crates a measurement histogram where the measurements have the same format as Qiskit regarding the qubit order and
    how multiple classical registers are formatted (e.g. "1 001").
    :param circuit:
    :param result:
    :param register_sizes:
    :return: histogram of the measurements in the same format as Qiskit
    """
    measurement_keys = circuit.all_measurement_key_names()
    register_measurements = []

    for register_name, size in reversed(
        register_sizes
    ):  # reverse order to have the same order as Qiskit
        for i in reversed(range(size)):  # reverse order to have the same order as Qiskit
            measurement_key = register_name + "_" + str(i)

            if measurement_key in measurement_keys:
                register_measurements.append(
                    result.measurements[measurement_key].reshape((-1))
                )
            else:
                register_measurements.append(np.zeros((result.repetitions,)))

    combined_measurements = np.stack(register_measurements, axis=1)
    unique_measurements, counts = np.unique(
        combined_measurements, return_counts=True, axis=0
    )

    histogram = {}

    for measurement, count in zip(unique_measurements, counts):
        register_measurement_strings = []
        index = 0

        for register_name, size in reversed(
            register_sizes
        ):  # reverse order to have the same order as Qiskit
            measurement_string = ""

            for _ in range(size):
                measurement_string += str(int(measurement[index]))
                index += 1

            register_measurement_strings.append(measurement_string)

        formatted_measurement_string = " ".join(register_measurement_strings)
        histogram[formatted_measurement_string] = count

    return histogram
