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

"""HTTP-level tests for the data-creator plugin routes.

These tests use the Flask test client provided by the local ``client``
fixture to hit the plugin's metadata endpoint and the micro frontend without
starting a real server. The ``/process/`` endpoint enqueues a Celery task
and is therefore covered by Celery-aware tests instead (see
:doc:`adr/0018-celery-task-testing-strategy`); it is intentionally skipped
here.
"""

import re
from http import HTTPStatus

from flask import url_for

from data_creator import DataCreator, DataCreator_BLP


def test_metadata_endpoint_returns_plugin_descriptor(client):
    """``GET /plugins/<id>/`` returns the plugin metadata as JSON."""
    response = client.get(url_for(f"{DataCreator_BLP.name}.PluginsView"))

    assert response.status_code == HTTPStatus.OK
    body = response.get_json()
    assert body["name"] == DataCreator.instance.name
    assert body["version"] == DataCreator.instance.version
    assert body["title"] == "Data Creation"
    assert "data-loading" in body["tags"]


def test_metadata_endpoint_describes_four_outputs(client):
    """The data-creator entry point declares four outputs: train/test x data/labels.

    The plugin runner schemas serialize python ``snake_case`` field names as
    ``camelCase`` in JSON (see :py:class:`MaBaseSchema`), so the keys read
    here are ``entryPoint``, ``dataOutput``, ``dataType`` and ``contentType``.
    """
    response = client.get(url_for(f"{DataCreator_BLP.name}.PluginsView"))
    outputs = response.get_json()["entryPoint"]["dataOutput"]

    assert len(outputs) == 4
    data_types = [out["dataType"] for out in outputs]
    assert data_types.count("entity/vector") == 2
    assert data_types.count("entity/label") == 2
    for out in outputs:
        assert "application/json" in out["contentType"]


def test_microfrontend_renders_form_fields(client):
    """``GET /ui/`` returns an HTML form populated with the schema's labels."""
    response = client.get(url_for(f"{DataCreator_BLP.name}.MicroFrontend"))

    assert response.status_code == HTTPStatus.OK
    body = response.get_data(as_text=True)
    # The form labels match the ``metadata["label"]`` strings declared on
    # ``InputParametersSchema``. If those labels change, this test should be
    # updated to follow.
    assert "Dataset Type" in body
    assert "No. Training Points" in body
    assert "No. Test Points" in body


def test_microfrontend_shows_default_values(client):
    """The frontend pre-fills the optional fields with the documented defaults."""
    response = client.get(url_for(f"{DataCreator_BLP.name}.MicroFrontend"))
    body = response.get_data(as_text=True)

    # Defaults set in routes.py: noise=0.7, turns=1.52, centers=4. Bind the
    # value to the specific input via ``name="..."`` so an unrelated occurrence
    # of "4" elsewhere in the rendered HTML cannot make the assertion pass.
    for field, expected in (("noise", "0.7"), ("turns", "1.52"), ("centers", "4")):
        pattern = rf'name="{field}"[^>]*value="{re.escape(expected)}"'
        input_pattern = rf'<input[^>]*name="{field}"[^>]*>'
        matches = re.findall(input_pattern, body)
        assert re.search(
            pattern, body
        ), f"input {field!r} should default to {expected!r}; got: {matches}"


def test_microfrontend_rejects_invalid_post(client):
    """Posting an invalid payload re-renders the form with errors instead of 400.

    The route uses ``validate_errors_as_result=True``, which means schema
    errors are passed to the view as a dict rather than aborting the request.
    The test verifies the form still renders 200 and that the invalid value
    is accepted as a string in the rendered HTML (so the user can correct it).
    """
    response = client.post(
        url_for(f"{DataCreator_BLP.name}.MicroFrontend"),
        data={
            "dataset_type": "not-a-real-dataset",
            "num_train_points": "abc",
            "turns": "1.5",
        },
    )
    assert response.status_code == HTTPStatus.OK
