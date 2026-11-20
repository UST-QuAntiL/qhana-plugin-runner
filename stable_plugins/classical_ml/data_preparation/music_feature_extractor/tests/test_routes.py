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

"""Tests for the micro frontend of the music feature extractor."""

import re
from http import HTTPStatus

from bs4 import BeautifulSoup
from flask import url_for

from music_feature_extractor import MusicFeatureExtractor_BLP
from music_feature_extractor.schemas import InputParametersSchema

EVENT_GROUP_FIELDS = (
    "include_melodic_intervals",
    "include_melodic_summary",
    "include_voice_activity",
)


def _micro_frontend(client) -> BeautifulSoup:
    """Return the parsed micro frontend page."""
    response = client.get(url_for(f"{MusicFeatureExtractor_BLP.name}.MicroFrontend"))
    assert response.status_code == HTTPStatus.OK, "The micro frontend must render."
    return BeautifulSoup(response.data, "html.parser")


def test_micro_frontend_renders_event_group_checkboxes(client):
    """Render a checkbox for every note-event feature group."""
    page = _micro_frontend(client)

    checkboxes = {
        field
        for field in EVENT_GROUP_FIELDS
        if (element := page.find("input", id=field)) and element.get("type") == "checkbox"
    }

    assert checkboxes == set(EVENT_GROUP_FIELDS), (
        "Every note-event group needs a checkbox. "
        "Otherwise the group cannot be selected in the UI."
    )


def test_template_places_every_schema_field(client):
    """List every schema field in the field placement of the template script."""
    page = _micro_frontend(client)
    script = "\n".join(tag.get_text() for tag in page.find_all("script"))

    placed = set(re.findall(r'\{ id: "(\w+)", target:', script))

    assert set(InputParametersSchema().fields) <= placed, (
        "The script removes the hidden field container after moving the fields. "
        "A field missing from fieldTargets silently disappears from the form."
    )
