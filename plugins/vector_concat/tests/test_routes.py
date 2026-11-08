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

import re
from http import HTTPStatus
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
import pytest
from flask import json, url_for

from plugins.vector_concat.schemas import ACCEPTED_CONTENT_TYPES
from vector_concat import VECTOR_CONCAT_BLP, VectorConcatPlugin


def _path(endpoint: str) -> str:
    return urlsplit(url_for(f"{VECTOR_CONCAT_BLP.name}.{endpoint}")).path


def test_metadata_endpoint_returns_full_descriptor(client):
    resp = client.get(url_for(f"{VECTOR_CONCAT_BLP.name}.PluginsView"))
    plugin = VectorConcatPlugin.instance

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_json()

    assert body["name"] == plugin.name
    assert body["version"] == plugin.version
    assert body["title"] == "Vector concationation plugin"
    assert body["description"] == plugin.description
    assert body["type"] == "processing"
    assert body["tags"] == plugin.tags

    entry_point = body["entryPoint"]
    assert entry_point["href"] == _path("CalcView")
    assert entry_point["uiHref"] == _path("MicroFrontend")
    assert entry_point["pluginDependencies"] == []

    assert entry_point["dataInput"] == [
        {
            "dataType": "entity/vector",
            "contentType": ACCEPTED_CONTENT_TYPES,
            "required": True,
            "parameter": "urls",
        }
    ]

    assert entry_point["dataOutput"] == [
        {
            "dataType": "entity/vector",
            "contentType": [
                "text/csv",
                "application/json",
                "application/X-lines+json",
            ],
            "required": True,
        },
        {
            "dataType": "entity/dimension-mapping",
            "contentType": ["application/json"],
            "required": True,
        },
    ]


def test_microfrontend_renders_form_fields(client):
    resp = client.get(url_for(f"{VECTOR_CONCAT_BLP.name}.MicroFrontend"))

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)

    assert "Data Sources" in body
    assert '<input type="hidden" name="urls" id="data_sources_input" value="">' in body

    soup = BeautifulSoup(body, "html.parser")

    output_format = soup.select_one("#output_format")
    assert output_format is not None
    selected_option = output_format.find("option", selected=True)
    assert selected_option is not None
    assert selected_option["value"] == "csv"

    option_values = [option["value"] for option in output_format.find_all("option")]
    assert option_values == ["csv", "json", "lines"]

    assert "Output File Suffix" in body
    assert 'name="outputSuffix"' in body

    assert f'formaction="{_path("CalcView")}"' in body
    assert 'acceptedInputType: "entity/vector"' in body

    script = soup.select_one("#accepted-content-types")
    assert script is not None
    assert json.loads(script.string) == [
        "text/csv",
        "application/json",
        "application/X-lines+json",
        "application/zip",
    ]


def test_microfrontend_post_echoes_submitted_output_format(client):
    resp = client.post(
        url_for(f"{VECTOR_CONCAT_BLP.name}.MicroFrontend"),
        data={"urls": "http://example.com/a.csv", "outputFormat": "json"},
    )

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)
    soup = BeautifulSoup(body, "html.parser")

    output_format = soup.select_one("#output_format")
    assert output_format is not None
    selected_option = output_format.find("option", selected=True)
    assert selected_option is not None
    assert selected_option["value"] == "json"


def test_microfrontend_invalid_post_rerenders_form_without_redirect(client):
    resp = client.post(
        url_for(f"{VECTOR_CONCAT_BLP.name}.MicroFrontend"),
        data={"urls": "not-a-url", "outputFormat": "csv"},
    )

    assert resp.status_code == HTTPStatus.OK
    body = resp.get_data(as_text=True)
    assert '<form id="form_main"' in body
    assert f'formaction="{_path("CalcView")}"' in body


@pytest.mark.parametrize(
    ("payload", "field", "message"),
    [
        ({"urls": "not-a-url"}, "urls", "Line 1: Not a valid URL."),
        ({}, "urls", "Missing data for required field."),
        (
            {"urls": "http://example.com/a.csv", "outputFormat": "xml"},
            "outputFormat",
            "Must be one of: csv, json, lines.",
        ),
        (
            {"urls": "http://example.com/a.csv", "outputSuffix": "bad name"},
            "outputSuffix",
            "Suffix may only contain letters, digits, '.', '_' and '-'.",
        ),
        (
            {"urls": "http://example.com/a.csv", "outputSuffix": "a/b"},
            "outputSuffix",
            "Suffix may only contain letters, digits, '.', '_' and '-'.",
        ),
    ],
)
def test_process_rejects_invalid_payload(client, payload, field, message):
    resp = client.post(url_for(f"{VECTOR_CONCAT_BLP.name}.CalcView"), data=payload)

    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert resp.get_json()["errors"]["form"][field] == [message]


def test_process_valid_payload_redirects_to_task(client):
    resp = client.post(
        url_for(f"{VECTOR_CONCAT_BLP.name}.CalcView"),
        data={"urls": "http://example.com/a.csv", "outputFormat": "csv"},
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert re.fullmatch(r"/tasks/\d+/", urlsplit(resp.headers["Location"]).path)


def test_process_accepts_valid_output_suffix(client):
    resp = client.post(
        url_for(f"{VECTOR_CONCAT_BLP.name}.CalcView"),
        data={
            "urls": "http://example.com/a.csv",
            "outputFormat": "csv",
            "outputSuffix": "run_42.v2",
        },
    )

    assert resp.status_code == HTTPStatus.SEE_OTHER
    assert re.fullmatch(r"/tasks/\d+/", urlsplit(resp.headers["Location"]).path)
