import json

import pytest

from .api_client import APIClient

test_data = [
    {
        "id": 10,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """,
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        h q[1];
        """,
        ],
        "dimHX": 1,
        "dimHR": 1,
        "qubit_intervals": [None, None],
        "expected": True,
    },
    {
        "id": 10,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """,
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[2];
        """,
        ],
        "dimHX": 1,
        "dimHR": 1,
        "qubit_intervals": [None, None],
        "expected": True,
    },
    {
        "id": 0,
        "qasmfilecontents": [
            """
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[6];
            h q[0];
            x q[2];
            """
        ],
        "dimHX": 2,
        "dimHR": 0,
        "qubit_intervals": [[[0, 1], [2, 3], [4, 5]]],
        "expected": True,
    },
    {
        "id": 1,
        "qasmfilecontents": [
            """
            OPENQASM 2.0;
            include \"qelib1.inc\";
            qreg q[4];
            h q[0];
            h q[1];
            h q[2];
            h q[3];
            """
        ],
        "dimHX": 2,
        "dimHR": 0,
        "qubit_intervals": [[[0, 1], [2, 3]]],
        "expected": True,
    },
    {
        "id": 2,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        h q[1];
        """,
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        """,
        ],
        "dimHX": 2,
        "dimHR": 1,
        "qubit_intervals": [None, None],
        "expected": False,
    },
    {
        "id": 3,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[6];
        h q[1];
        """
        ],
        "dimHX": 2,
        "dimHR": 1,
        "qubit_intervals": [[[0, 2], [3, 5]]],
        "expected": False,
    },
    {
        "id": 4,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        h q[1];
        """,
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        """,
            """
        OPENQASM 2.0;
        include \"qelib1.inc\";
        qreg q[3];
        h q[1];
        cx q[1], q[0];
        """,
        ],
        "dimHX": 2,
        "dimHR": 1,
        "qubit_intervals": [None, None, None],
        "expected": True,
    },
    {
        "id": 5,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[1];
        """,
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        """,
        ],
        "dimHX": 2,
        "dimHR": 1,
        "qubit_intervals": [None, None],
        "expected": False,
    },
    {
        "id": 6,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[1];
        """,
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        """,
        ],
        "dimHX": 2,
        "dimHR": 1,
        "qubit_intervals": [[[0, 2]], [[0, 2]]],
        "expected": False,
    },
    {
        "id": 7,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        x q[1];
        """,
        ],
        "dimHX": 2,
        "dimHR": 1,
        "qubit_intervals": [[[0, 2], [3, 5]]],
        "expected": False,
    },
    {
        "id": 8,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q[0];
        h q[1];
        ccx q[0], q[1], q[2];
        """,
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        """,
        ],
        "dimHX": 2,
        "dimHR": 1,
        "qubit_intervals": [None, None],
        "expected": False,
    },
    {
        "id": 9,
        "qasmfilecontents": [
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        h q[0];
        h q[1];
        ccx q[0], q[1], q[2];
        """,
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        """,
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[2];
        """,
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[0];
        """,
            """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        x q[0];
        x q[1];
        x q[2];
        """,
        ],
        "dimHX": 2,
        "dimHR": 1,
        "qubit_intervals": [None, None, None, None, None],
        "expected": True,
    },
]


@pytest.fixture
def file_maker_client():
    return APIClient(
        base_url="http://localhost:5005", plugin_name="filesMaker", version="v0-0-1"
    )


@pytest.fixture
def plugin_to_test_client():
    return APIClient(
        base_url="http://localhost:5005",
        plugin_name="linear_dependence_HX_classical",
        version="v0-0-1",
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


def extract_result(response):
    log_data = response.get("log", "{}")
    try:
        log_dict = json.loads(log_data)
    except json.JSONDecodeError:
        pytest.fail("Failed to parse JSON response from API")
    assert isinstance(log_dict, dict), "Response log should be a dictionary"
    assert "result" in log_dict, "Response should contain 'result' key"
    return log_dict["result"]


@pytest.mark.parametrize("case", test_data, ids=[f"case_{d['id']}" for d in test_data])
def test_api_request(case, file_maker_client, plugin_to_test_client):
    urls_and_metadata = []
    for qasm, metadata in zip(case["qasmfilecontents"], case["qubit_intervals"]):
        qasm_payload = {"qasmCode": qasm}
        file_maker_response = file_maker_client.send_request(qasm_payload)
        qasm_url = extract_qasm_url(file_maker_response)
        if metadata:
            urls_and_metadata.append([qasm_url, {"qubit_intervals": metadata}])
        else:
            urls_and_metadata.append([qasm_url])

    payload = {
        "qasmInputList": json.dumps(urls_and_metadata),
        "dimHX": case["dimHX"],
        "dimHR": case["dimHR"],
    }
    response = plugin_to_test_client.send_request(payload, max_retries=15)
    result = extract_result(response)

    assert (
        result == case["expected"]
    ), f"Test case {case['id']} failed: Expected {case['expected']}, got {result}"
