import json

import pytest
from common.plugin_utils.task_util import generate_one_circuit_with_multiple_states_weak
from qiskit import QuantumCircuit

from .api_client import APIClient

testdata = [
    {
        "id": 0,
        "COPIES_PER_STATE": 1,
        "qasm1": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "interval1": [[3, 3]],
        "qasm2": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "interval2": [[3, 3]],
        "expected_qasm": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[8];
        x q[3];
        x q[7];
        """,
        "expected_interval": [[3, 3], [7, 7]],
    },
    {
        "id": 1,
        "COPIES_PER_STATE": 1,
        "qasm1": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[2];
        """,
        "interval1": [[1, 2]],
        "qasm2": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[2];
        """,
        "interval2": [[1, 2]],
        "expected_qasm": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[8];
        x q[2];
        x q[6];
        """,
        "expected_interval": [[1, 2], [5, 6]],
    },
    {
        "id": 2,
        "COPIES_PER_STATE": 2,
        "qasm1": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "interval1": [[3, 3]],
        "qasm2": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[3];
        """,
        "interval2": [[3, 3]],
        "expected_qasm": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[16];
        x q[3];
        x q[7];
        x q[11];
        x q[15];
        """,
        "expected_interval": [[3, 3], [7, 7], [11, 11], [15, 15]],
    },
    {
        "id": 3,
        "COPIES_PER_STATE": 2,
        "qasm1": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[2];
        """,
        "interval1": [[1, 2]],
        "qasm2": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        x q[2];
        """,
        "interval2": [[1, 2]],
        "expected_qasm": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[16];
        x q[2];
        x q[6];
        x q[10];
        x q[14];
        """,
        "expected_interval": [[1, 2], [5, 6], [9, 10], [13, 14]],
    },
    {
        "id": 0,
        "COPIES_PER_STATE": 4,
        "qasm1": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[0];
        """,
        "interval1": [[0, 1]],
        "qasm2": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        x q[1];
        """,
        "interval2": [[0, 1]],
        "expected_qasm": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[16];
        x q[0];
        x q[3];
        x q[4];
        x q[7];
        x q[8];
        x q[11];
        x q[12];
        x q[15];
        """,
        "expected_interval": [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11],
            [12, 13],
            [14, 15],
        ],
    },
]


@pytest.fixture
def file_maker_client():
    return APIClient(
        base_url="http://localhost:5005", plugin_name="filesMaker", version="v0-0-1"
    )


def extract_qasm_url(response):
    assert isinstance(response, dict), "Response should be a dictionary"
    assert "outputs" in response, "Response should contain 'outputs' key"
    outputs = response.get("outputs", [])
    assert isinstance(outputs, list) and outputs, "'outputs' should be a non-empty list"
    qasm_url = next(
        (item.get("href") for item in outputs if item.get("name") == "circuit.qasm"), None
    )
    assert qasm_url, "QASM file URL should not be None"
    return qasm_url


@pytest.mark.parametrize("case", testdata, ids=[f"case_{d['id']}" for d in testdata])
def test_generate_one_circuit_with_multiple_states_weak(case, file_maker_client):
    """
    Parameterized test for generate_one_circuit_with_two_states with real QASM file URLs.
    """
    # Prepare payloads for QASM file generation
    qasm_payload1 = {"qasmCode": case["qasm1"]}
    qasm_payload2 = {"qasmCode": case["qasm2"]}

    # Send request to generate QASM file URLs
    file_response1 = file_maker_client.send_request(qasm_payload1)
    file_response2 = file_maker_client.send_request(qasm_payload2)

    qasm_url1 = extract_qasm_url(file_response1)
    qasm_url2 = extract_qasm_url(file_response2)

    # Prepare QASM input list with URLs1
    qasm_input_list = [
        (qasm_url1, case["interval1"]),
        (qasm_url2, case["interval2"]),
    ]

    COPIES_PER_STATE = int(case["COPIES_PER_STATE"])

    # Execute function
    merged_circuit, new_intervals = generate_one_circuit_with_multiple_states_weak(
        qasm_input_list, COPIES_PER_STATE
    )

    # Convert expected result into a QuantumCircuit
    expected_circuit = QuantumCircuit.from_qasm_str(case["expected_qasm"])

    assert (
        merged_circuit.qasm() == expected_circuit.qasm()
    ), "The merged QASM circuits do not match!"

    # Check if the qubit intervals are correctly updated
    expected_intervals = case["expected_interval"]
    assert (
        new_intervals == expected_intervals
    ), f"Expected intervals {expected_intervals}, but got {new_intervals}"
