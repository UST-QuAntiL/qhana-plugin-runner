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

from http import HTTPStatus

import pytest
from bs4 import BeautifulSoup
from flask import url_for

from feature_engineering_pipeline import FEATURE_ENGINEERING_PIPELINE_BLP
from feature_engineering_pipeline.tests.data import make_router_task


def render(client, endpoint: str, data: dict | None = None, **kwargs) -> BeautifulSoup:
    url = url_for(f"{FEATURE_ENGINEERING_PIPELINE_BLP.name}.{endpoint}", **kwargs)
    resp = client.get(url) if data is None else client.post(url, data=data)
    assert resp.status_code == HTTPStatus.OK
    return BeautifulSoup(resp.get_data(as_text=True), "html.parser")


def open_sections(soup: BeautifulSoup) -> set:
    return {
        section.summary.get_text(strip=True)
        for section in soup.select("details.settings-section")
        if section.has_attr("open")
    }


def test_first_step_defaults(client):
    soup = render(client, "MicroFrontend")

    assert soup.select_one("#mds_dimensions")["value"] == "1"
    assert soup.select_one("#concat_output").has_attr("checked")


def test_first_step_opens_only_the_basic_and_concatenation_sections(client):
    soup = render(client, "MicroFrontend")

    assert open_sections(soup) == {"Basic Data", "Vector Concatenation Settings"}


def test_first_step_keeps_an_unchecked_concatenation_after_validation(client):
    # A submitted form omits the unchecked checkbox, but always contains the entities url.
    soup = render(client, "MicroFrontend", data={"entitiesUrl": ""})

    assert not soup.select_one("#concat_output").has_attr("checked")


@pytest.mark.parametrize("concat_output,expanded", [(True, True), (False, False)])
def test_routing_step_opens_the_pca_section_for_a_concatenated_output(
    client, concat_output, expanded
):
    db_task = make_router_task(concatOutput=concat_output)

    soup = render(client, "RoutingStepFrontend", db_id=db_task.id)

    assert soup.select_one("#pca-settings").has_attr("open") is expanded
