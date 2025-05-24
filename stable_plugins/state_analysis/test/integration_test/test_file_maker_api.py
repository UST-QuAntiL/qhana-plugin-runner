import pytest

from .api_client import APIClient


@pytest.fixture
def api_client():
    """Fixture to create an APIClient instance."""
    return APIClient(
        base_url="http://localhost:5005", plugin_name="filesMaker", version="v0-0-1"
    )


def test_send_request(api_client):
    """Test APIClient's send_request method with valid data."""
    data = {
        "qasmCode": 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nx q[0];',
    }

    try:
        response = api_client.send_request(data)
        assert isinstance(response, dict), "Response should be a dictionary"
    except ValueError as e:
        pytest.fail(f"API request failed: {e}")
