import json

import pytest

from .api_client import APIClient
from .vectorFormatter import format_complex_vectors

test_data = [
    {
        "id": 1,
        "vectors": [[1, 0, 0, 0], [1, 0, 0, 0]],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 1,
        "vectors": [[1, 0, 0, 0]],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 1,
        "vectors": [[0, 0, 0, 0]],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 1,
        "vectors": [
            [1j, 1],
            [0, 1e-11],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 2,
        "vectors": [
            [1 + 1j, 2 - 1j],
            [2 - 1j, -1 - 1j],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 3,
        "vectors": [
            [1, 1j, 0],
            [1j, 0, 1],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 4,
        "vectors": [
            [1 + 1j, 1 - 1j],
            [1 - 1j, 1 + 1j],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 5,
        "vectors": [
            [0, 0, 1],
            [0, 0, 1e-12],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 6,
        "vectors": [
            [2 + 3j, -1j],
            [-3 + 2j, 1j],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 7,
        "vectors": [
            [1, 0, 1j],
            [1j, 1, 0],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 8,
        "vectors": [
            [0, 1 + 1j, 1 - 1j],
            [1 - 1j, 0, 1 + 1j],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 9,
        "vectors": [
            [1j, 1j, 1j],
            [-1j, -1j, -1j],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 10,
        "vectors": [
            [1e-12, 1j, 0],
            [1j, 0, 1e-12],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 11,
        "vectors": [
            [1 + 1j, 1 + 1j],
            [1 - 1j, 1 - 1j],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 12,
        "vectors": [
            [2j, 3j, 4j],
            [-2j, -3j, -4j],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 13,
        "vectors": [
            [1, 1, 1],
            [-1, -1, -1],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 14,
        "vectors": [
            [1j, 1, 0],
            [0, 1j, 1],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 15,
        "vectors": [
            [1 + 1j, 0, 0],
            [0, 1 - 1j, 0],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 16,
        "vectors": [
            [0, 1 + 1j, 0],
            [0, 0, 1 - 1j],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 17,
        "vectors": [
            [1, 1, 1j],
            [1j, 1, 1],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 18,
        "vectors": [
            [1j, 1, 1],
            [1, 1j, 1],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 19,
        "vectors": [
            [1, 1j, 1],
            [1, 1, 1j],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 20,
        "vectors": [
            [1e-11, 1j, 1],
            [1j, 0, 1e-11],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 21,
        "vectors": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 22,
        "vectors": [
            [1, 2, 3],
            [2, 4, 6],
            [1, 1, 1],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 23,
        "vectors": [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 24,
        "vectors": [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 25,
        "vectors": [
            [1],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 26,
        "vectors": [
            [0],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 27,
        "vectors": [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 28,
        "vectors": [
            [1, 0, 0, 0, 0, 0, 0, 0],  # e1
            [0, 1, 0, 0, 0, 0, 0, 0],  # e2
            [0, 0, 1, 0, 0, 0, 0, 0],  # e3
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 29,
        "vectors": [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        "tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 30,
        "vectors": [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 31,
        "vectors": [
            [0, 0],
            [0, 0],
        ],
        "tolerance": None,
        "expected": True,
    },
    {
        "id": 32,
        "vectors": [
            [1, 1],
            [1, -1],
        ],
        "tolerance": None,
        "expected": False,
    },
]


@pytest.fixture
def plugin_to_test_client():
    return APIClient(
        base_url="http://localhost:5005",
        plugin_name="classical_lineardependence",
        version="v0-0-1",
    )


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
def test_api_request(case, plugin_to_test_client):
    """Parameterized test for APIClient using testdata cases."""
    formated_vectors = format_complex_vectors(case["vectors"])
    payload = {
        "vectors": json.dumps(formated_vectors),
        "tolerance": case["tolerance"],
    }

    response = plugin_to_test_client.send_request(payload)
    result = extract_result(response)

    assert (
        result == case["expected"]
    ), f"Test case {case['id']} failed: Expected {case['expected']}, got {result}"
