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

"""Tests for the dimension mapping support of the csv visualization plugin.

The plugin is a single module rather than a package, so it cannot host its
own test package. The tests use the repo-root layout permitted by ADR-0019.
"""

from http import HTTPStatus

import pytest
from flask import url_for
from requests.exceptions import HTTPError

from tests.utils import MockResponse

MAPPING_URL = "http://example.com/concatenated_dimension_mapping.json"

MAPPING = [
    {
        "ID": "dim0",
        "href": "",
        "inputIndex": 0,
        "source": "color.json",
        "sourceUrl": "http://example.com/vectors.zip",
        "zipMember": "color.json",
        "sourceDimension": "dim0",
    },
    {
        "ID": "dim1",
        "href": "",
        "inputIndex": 0,
        "source": "color.json",
        "sourceUrl": "http://example.com/vectors.zip",
        "zipMember": "color.json",
        "sourceDimension": "dim1",
    },
    {
        "ID": "dim2",
        "href": "",
        "inputIndex": 1,
        "source": "shape.json",
        "sourceUrl": "http://example.com/vectors.zip",
        "zipMember": "shape.json",
        "sourceDimension": "dim0",
    },
]


@pytest.fixture(scope="module")
def blueprint_name(app):
    """Name of the plugin blueprint, used to build the endpoint names.

    ``create_app`` adds the plugin folders to ``sys.path``, so the plugin
    module can only be imported once the ``app`` fixture has run.
    """
    from csv_visualization import CSV_BLP

    return CSV_BLP.name


@pytest.fixture
def mapping_file(monkeypatch):
    """Serve the dimension mapping from memory."""

    def mock_open_url(url, *args, **kwargs):
        if url != MAPPING_URL:
            raise HTTPError(f"Unknown url {url}")
        return MockResponse(url, "application/json", json_data=MAPPING)

    monkeypatch.setattr(
        "qhana_plugin_runner.plugin_utils.dimension_mapping.open_url", mock_open_url
    )


def test_metadata_lists_the_dimension_mapping_input(client, blueprint_name):
    resp = client.get(url_for(f"{blueprint_name}.PluginsView"))

    assert resp.status_code == HTTPStatus.OK
    assert resp.get_json()["entryPoint"]["dataInput"] == [
        {
            "dataType": "*",
            "contentType": ["text/csv"],
            "required": True,
            "parameter": "data",
        },
        {
            "dataType": "entity/dimension-mapping",
            "contentType": ["application/json"],
            "required": False,
            "parameter": "dimensionMappingUrl",
        },
    ]


def test_microfrontend_renders_the_dimension_mapping_field(client, blueprint_name):
    resp = client.get(url_for(f"{blueprint_name}.MicroFrontend"))

    assert resp.status_code == HTTPStatus.OK
    assert 'name="dimensionMappingUrl"' in resp.get_data(as_text=True)


def test_dimension_labels_name_the_feature_of_every_dimension(
    client, blueprint_name, mapping_file
):
    resp = client.get(
        url_for(f"{blueprint_name}.get_dimension_labels", dimensionMappingUrl=MAPPING_URL)
    )

    assert resp.status_code == HTTPStatus.OK
    assert resp.get_json() == {
        "dim0": "color",
        "dim1": "color",
        "dim2": "shape",
    }


def test_dimension_labels_are_empty_without_a_mapping(client, blueprint_name):
    resp = client.get(url_for(f"{blueprint_name}.get_dimension_labels"))

    assert resp.status_code == HTTPStatus.OK
    assert resp.get_json() == {}


def test_dimension_labels_reject_an_unreachable_mapping(
    client, blueprint_name, mapping_file
):
    resp = client.get(
        url_for(
            f"{blueprint_name}.get_dimension_labels",
            dimensionMappingUrl="http://example.com/missing.json",
        )
    )

    assert resp.status_code == HTTPStatus.BAD_REQUEST
