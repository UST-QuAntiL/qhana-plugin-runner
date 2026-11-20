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

"""Tests for the feature group selection of the input schema."""

import pytest

from music_feature_extractor.event_features import EVENT_GROUP_ORDER
from music_feature_extractor.feature_groups import SELECTABLE_GROUP_ORDER
from music_feature_extractor.schemas import (
    FEATURE_GROUP_ORDER,
    FEATURE_PRESET_GROUPS,
    resolve_feature_selection,
    resolve_feature_selection_from_values,
)


def test_schema_group_order_matches_feature_groups():
    """Keep the group order of the schema and the vector builder in sync."""
    assert FEATURE_GROUP_ORDER == SELECTABLE_GROUP_ORDER, (
        "schemas.py and feature_groups.py list the groups separately. "
        "A group missing in one of them could be selected but never computed, "
        "or computed but never selected."
    )


@pytest.mark.parametrize("preset", ["basic", "extended", "full"])
def test_presets_exclude_event_groups(preset):
    """Keep the existing presets free of the note-event groups."""
    included = set(FEATURE_PRESET_GROUPS[preset]) & set(EVENT_GROUP_ORDER)

    assert not included, (
        f"The '{preset}' preset must produce the same vectors as before. "
        "The note-event groups are therefore only available with the custom preset."
    )


def test_presets_ignore_event_group_flags():
    """Ignore note-event checkboxes unless the custom preset is used."""
    selection = resolve_feature_selection(
        feature_preset="full",
        include_melodic_intervals=True,
        include_voice_activity=True,
    )

    assert not any(selection[group] for group in EVENT_GROUP_ORDER), (
        "Checkbox flags only apply to the custom preset. "
        "The full preset keeps its original groups."
    )


def test_custom_preset_enables_event_groups():
    """Enable the note-event groups from the custom preset form values."""
    _, selection = resolve_feature_selection_from_values(
        {
            "featurePreset": "custom",
            "includeMelodicIntervals": "true",
            "includeMelodicSummary": "true",
            "includeVoiceActivity": "true",
        }
    )

    assert all(selection[group] for group in EVENT_GROUP_ORDER), (
        "The micro frontend submits the checkboxes as strings. "
        "All three note-event checkboxes must be read."
    )
