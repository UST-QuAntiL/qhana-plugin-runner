import json

import pytest

from .api_client import APIClient

test_data = [
    {
        "id": 0,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[1];
        cx q[1],q[0];

        """,
        "qubit_interval": None,
        "dimHX": 1,
        "dimHR": 1,
        "tolerance": 1e-10,
        "expected": 2,
    },
    {
        "id": 1,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        """,
        "qubit_interval": None,
        "dimHX": 1,
        "dimHR": 0,
        "tolerance": None,
        "expected": 1,
    },
    {
        "id": 2,
        "qasmfilecontent": """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];

        """,
        "qubit_interval": None,
        "dimHX": 1,
        "dimHR": 1,
        "tolerance": None,
        "expected": 1,
    },
]


@pytest.fixture
def file_maker_client():
    """Fixture to create an APIClient instance for the FileMaker plugin."""
    return APIClient(
        base_url="http://localhost:5005", plugin_name="filesMaker", version="v0-0-1"
    )


@pytest.fixture
def plugin_to_test_client():
    return APIClient(
        base_url="http://localhost:5005",
        plugin_name="schmidt_rank_classical",
        version="v0-0-1",
    )


def extract_qasm_url(response):
    """Extracts the QASM file URL from the API response."""
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
    """Extracts the result from the API response."""
    log_data = response.get("log", "{}")
    try:
        log_dict = json.loads(log_data)
    except json.JSONDecodeError:
        pytest.fail("Failed to parse JSON response from API")

    assert isinstance(log_dict, dict), "response log should be a dictionary"
    assert "result" in log_dict, "Response should contain 'result' key"

    return log_dict["result"]


@pytest.mark.parametrize("case", test_data, ids=[f"case_{d['id']}" for d in test_data])
def test_api_request(case, file_maker_client, plugin_to_test_client):
    """Parameterized test for APIClient using testdata cases."""
    qasm_payload = {
        "qasmCode": case["qasmfilecontent"],
    }

    file_maker_response = file_maker_client.send_request(qasm_payload)
    qasm_url = extract_qasm_url(file_maker_response)

    payload = {
        "dimHX": case["dimHX"],
        "dimHR": case["dimHR"],
        "tolerance": case["tolerance"],
    }

    if case["qubit_interval"]:
        payload["qasmInputList"] = json.dumps(
            [[qasm_url, {"qubit_intervals": case["qubit_interval"]}]]
        )
    else:
        payload["qasmInputList"] = json.dumps([[qasm_url]])

    response = plugin_to_test_client.send_request(payload)
    result = extract_result(response)

    assert (
        result == case["expected"]
    ), f"Test case {case['id']} failed: Expected {case['expected']}, got {result}"
