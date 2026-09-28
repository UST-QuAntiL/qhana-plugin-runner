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

"""Tests for the micro frontend and the process endpoint of the normalization plugin."""

from http import HTTPStatus
from urllib.parse import urlsplit

from flask import url_for

from .. import NORMALIZATION_BLP

ENTITIES_URL = "http://example.com/entities.json"

VALID_FORM = {
    "entitiesUrl": ENTITIES_URL,
    "attributes": "x",
    "outputRangeMin": 0.0,
    "outputRangeMax": 1.0,
}


def _path(endpoint: str) -> str:
    return urlsplit(url_for(f"{NORMALIZATION_BLP.name}.{endpoint}")).path


def test_microfrontend_renders_client_side_range_validation(client):
    resp = client.get(url_for(f"{NORMALIZATION_BLP.name}.MicroFrontend"))

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)
    assert f'formaction="{_path("ProcessView")}"' in body
    # the client side check must block submit before the request produces a 422
    assert "setCustomValidity" in body
    assert "input#output_range_start" in body
    assert "input#input_range_max" in body


def test_microfrontend_reports_equal_output_bounds_per_field(client):
    resp = client.post(
        url_for(f"{NORMALIZATION_BLP.name}.MicroFrontend"),
        data={**VALID_FORM, "outputRangeMin": 1.0, "outputRangeMax": 1.0},
    )

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)
    assert (
        body.count("The output range start and end must not be equal (both are 1.0).")
        == 2
    )


def test_process_still_rejects_equal_output_bounds(client):
    """The schema stays the safety net for direct API calls."""
    resp = client.post(
        url_for(f"{NORMALIZATION_BLP.name}.ProcessView"),
        data={**VALID_FORM, "outputRangeMin": 1.0, "outputRangeMax": 1.0},
    )

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    message = "The output range start and end must not be equal (both are 1.0)."
    errors = resp.get_json()["errors"]["form"]
    assert errors["outputRangeMin"] == [message]
    assert errors["outputRangeMax"] == [message]
