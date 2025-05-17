import json

import pytest

from .api_client import APIClient
from .vectorFormatter import format_complex_vectors

test_data = [
    {
        "id": 1,
        "vectors": [
            [
                0.7071067811865475,
                0.0,
                0.0,
                0.7071067811865475,
            ]
        ],
        "dimHX": 1,
        "dimHR": 1,
        "tolerance": 1e-10,
        "expected": 2,
    },
    {
        "id": 2,
        "vectors": [
            [
                1.0,
                0.0,
            ]
        ],
        "dimHX": 1,
        "dimHR": 0,
        "tolerance": None,
        "expected": 1,
    },
    {
        "id": 3,
        "vectors": [
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        ],
        "dimHX": 1,
        "dimHR": 1,
        "tolerance": None,
        "expected": 1,
    },
]


@pytest.fixture
def plugin_to_test_client():
    return APIClient(
        base_url="http://localhost:5005",
        plugin_name="schmidt_rank_classical",
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
    formated_vector = format_complex_vectors(case["vectors"])
    payload = {
        "vector": json.dumps(formated_vector[0]),
        "dimHX": case["dimHX"],
        "dimHR": case["dimHR"],
        "tolerance": case["tolerance"],
    }

    response = plugin_to_test_client.send_request(payload, max_retries=20)
    result = extract_result(response)

    assert (
        result == case["expected"]
    ), f"Test case {case['id']} failed: Expected {case['expected']}, got {result}"
