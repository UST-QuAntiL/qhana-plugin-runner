import json

import pytest

from .api_client import APIClient
from .vectorFormatter import format_complex_vectors

test_data = [
    {
        "id": 1,
        "vector1": [1j, 1],
        "vector2": [0, 1e-11],
        "innerproduct_tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 2,
        "vector1": [1 + 1j, 2 - 1j],
        "vector2": [2 - 1j, -1 - 1j],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 3,
        "vector1": [1, 1j, 0],
        "vector2": [1j, 0, 1],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 4,
        "vector1": [1 + 1j, 1 - 1j],
        "vector2": [1 - 1j, 1 + 1j],
        "innerproduct_tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 5,
        "vector1": [1j, 0.1],
        "vector2": [0, 1e-12],
        "innerproduct_tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 6,
        "vector1": [2 + 3j, -1j],
        "vector2": [-3 + 2j, 1j],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 7,
        "vector1": [1, 0, 1j],
        "vector2": [1j, 1, 0],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 8,
        "vector1": [0, 1 + 1j, 1 - 1j],
        "vector2": [1 - 1j, 0, 1 + 1j],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 9,
        "vector1": [1j, 1j, 1j],
        "vector2": [-1j, -1j, -1j],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 10,
        "vector1": [1e-12, 1j, 0],
        "vector2": [1j, 0, 1e-12],
        "innerproduct_tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 11,
        "vector1": [1 + 1j, 1 + 1j],
        "vector2": [1 - 1j, 1 - 1j],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 12,
        "vector1": [2j, 3j, 4j],
        "vector2": [-2j, -3j, -4j],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 13,
        "vector1": [1, 1, 1],
        "vector2": [-1, -1, -1],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 14,
        "vector1": [1j, 1, 0],
        "vector2": [0, 1j, 1],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 15,
        "vector1": [1 + 1j, 0, 0],
        "vector2": [0, 1 - 1j, 0],
        "innerproduct_tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 16,
        "vector1": [0, 1 + 1j, 0],
        "vector2": [0, 0, 1 - 1j],
        "innerproduct_tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 17,
        "vector1": [1, 1, 1j],
        "vector2": [1j, 1, 1],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 18,
        "vector1": [1j, 1, 1],
        "vector2": [1, 1j, 1],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 19,
        "vector1": [1, 1j, 1],
        "vector2": [1, 1, 1j],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 20,
        "vector1": [1e-11, 1j, 1],
        "vector2": [1j, 0, 1e-11],
        "innerproduct_tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 21,
        "vector1": [1, 0, 0],
        "vector2": [0, 1, 0],
        "innerproduct_tolerance": None,
        "expected": True,
    },
    {
        "id": 22,
        "vector1": [1, 1, 1],
        "vector2": [1, 1, 1],
        "innerproduct_tolerance": None,
        "expected": False,
    },
    {
        "id": 23,
        "vector1": [0, 1],
        "vector2": [0, 1],
        "innerproduct_tolerance": 1e-10,
        "expected": False,
    },
    {
        "id": 24,
        "vector1": [1, 0],
        "vector2": [1, 0],
        "innerproduct_tolerance": None,
        "expected": False,
    },
]


@pytest.fixture
def plugin_to_test_client():
    return APIClient(
        base_url="http://localhost:5005",
        plugin_name="classical_orthogonality",
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
    formated_vectors = format_complex_vectors([case["vector1"], case["vector2"]])
    payload = {
        "vectors": json.dumps(formated_vectors),
        "tolerance": case["innerproduct_tolerance"],
    }

    response = plugin_to_test_client.send_request(payload)
    result = extract_result(response)

    assert (
        result == case["expected"]
    ), f"Test case {case['id']} failed: Expected {case['expected']}, got {result}"
