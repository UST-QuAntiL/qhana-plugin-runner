import json

import pytest

from .api_client import APIClient
from .vectorFormatter import format_complex_vectors

test_data = [
    {
        "id": 0,
        "vectors": [[1, 0, 0], [0, 0, 1]],
        "innerproduct_tolerance": 1e-10,
        "expected": True,
    },
    {
        "id": 1,
        "vectors": [[1e-5, 1j, 0], [1j, 0, 1e-12]],
        "innerproduct_tolerance": 1e-3,
        "expected": True,
    },
    {
        "id": 2,
        "vectors": [[1e-5, 1j, 0], [1j, 0, 1e-12]],
        "innerproduct_tolerance": 0,
        "expected": False,
    },
    {
        "id": 3,
        "vectors": [[1, 0, 0], [1, 1, 1], [0, 1, 0], [0, 0, 1]],
        "innerproduct_tolerance": 0,
        "expected": False,
    },
    {
        "id": 4,
        "vectors": [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]],
        "innerproduct_tolerance": 0,
        "expected": True,
    },
    {
        "id": 5,
        "vectors": [[1, 0, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "innerproduct_tolerance": 0,
        "expected": False,
    },
    {
        "id": 6,
        "vectors": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "innerproduct_tolerance": 0,
        "expected": True,
    },
    {
        "id": 7,
        "vectors": [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
        "innerproduct_tolerance": 0,
        "expected": False,
    },
]


@pytest.fixture
def plugin_to_test_client():
    return APIClient(
        base_url="http://localhost:5005",
        plugin_name="pairwise_orthogonality_classical",
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
        "tolerance": case["innerproduct_tolerance"],
    }

    response = plugin_to_test_client.send_request(payload)
    result = extract_result(response)

    assert (
        result == case["expected"]
    ), f"Test case {case['id']} failed: Expected {case['expected']}, got {result}"
