# Copyright 2026 QHAna plugin runner contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from .api_client import APIClient


@pytest.fixture
def api_client():
    """Fixture to create an APIClient instance."""
    return APIClient(plugin_name="filesMaker", version="v0-0-1")


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
