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

from dataclasses import dataclass
from typing import Mapping

import marshmallow as ma
from marshmallow import post_load, validate

from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema, MaBaseSchema

from .io_utils import DECLARED_FORMAT_OPTIONS, SOURCE_MODE_OPTIONS

FEATURE_PRESETS = ("basic", "extended", "full", "custom")
FEATURE_GROUP_ORDER = (
    "pitch_stats",
    "intervals",
    "rhythm",
    "meter_tempo",
    "texture",
    "dynamics",
    "tonality",
    "harmony",
)
FEATURE_PRESET_GROUPS = {
    "basic": (),
    "extended": (
        "pitch_stats",
        "intervals",
        "rhythm",
        "meter_tempo",
        "texture",
        "dynamics",
        "tonality",
    ),
    "full": FEATURE_GROUP_ORDER,
    "custom": (),
}


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class InputParameters:
    source_url: str
    source_mode: str = "auto"
    declared_format: str = "auto"
    max_files: int = 1000
    continue_on_error: bool = True
    feature_preset: str = "basic"
    include_pitch_stats: bool = False
    include_intervals: bool = False
    include_rhythm: bool = False
    include_meter_tempo: bool = False
    include_texture: bool = False
    include_dynamics: bool = False
    include_tonality: bool = False
    include_harmony: bool = False

    def __str__(self):
        return str(self.__dict__)


class InputParametersSchema(FrontendFormBaseSchema):
    source_url = FileUrl(
        required=True,
        allow_none=False,
        data_key="sourceUrl",
        data_input_type="*",
        data_content_types=[
            "application/zip",
            "application/xml",
            "text/xml",
            "application/vnd.recordare.musicxml+xml",
            "audio/midi",
            "audio/x-midi",
            "application/octet-stream",
        ],
        metadata={
            "label": "Source URL",
            "description": "URL to a single MusicXML/MXL/MIDI file or a ZIP archive with music files.",
            "input_type": "text",
        },
    )
    source_mode = ma.fields.String(
        required=False,
        allow_none=False,
        data_key="sourceMode",
        load_default="auto",
        validate=validate.OneOf(SOURCE_MODE_OPTIONS),
        metadata={
            "label": "Source Mode",
            "description": "How to interpret the input source: auto-detect, single file, or ZIP batch.",
            "input_type": "select",
            "options": {mode: mode for mode in SOURCE_MODE_OPTIONS},
        },
    )
    declared_format = ma.fields.String(
        required=False,
        allow_none=False,
        data_key="declaredFormat",
        load_default="auto",
        validate=validate.OneOf(DECLARED_FORMAT_OPTIONS),
        metadata={
            "label": "Declared Format",
            "description": "Optional explicit format. Keep auto for mixed ZIP batches.",
            "input_type": "select",
            "options": {fmt: fmt for fmt in DECLARED_FORMAT_OPTIONS},
        },
    )
    max_files = ma.fields.Integer(
        required=False,
        allow_none=False,
        data_key="maxFiles",
        load_default=1000,
        validate=validate.Range(min=1),
        metadata={
            "label": "Max Files",
            "description": "Maximum number of files processed when source mode resolves to ZIP.",
            "input_type": "number",
        },
    )
    continue_on_error = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="continueOnError",
        load_default=True,
        metadata={
            "label": "Continue On Error",
            "description": "Continue processing remaining files if a file fails extraction.",
            "input_type": "checkbox",
        },
    )
    feature_preset = ma.fields.String(
        required=False,
        allow_none=False,
        data_key="featurePreset",
        load_default="basic",
        validate=validate.OneOf(FEATURE_PRESETS),
        metadata={
            "label": "Feature Preset",
            "description": "Choose between baseline, extended, full, or custom feature groups.",
            "input_type": "select",
            "options": {preset: preset for preset in FEATURE_PRESETS},
        },
    )
    include_pitch_stats = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="includePitchStats",
        load_default=False,
        metadata={
            "label": "Include Pitch Stats",
            "description": "Adds pitch entropy, variety, and normalized pitch-range statistics.",
            "input_type": "checkbox",
        },
    )
    include_intervals = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="includeIntervals",
        load_default=False,
        metadata={
            "label": "Include Intervals",
            "description": "Adds melodic interval histogram and interval ratios.",
            "input_type": "checkbox",
        },
    )
    include_rhythm = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="includeRhythm",
        load_default=False,
        metadata={
            "label": "Include Rhythm",
            "description": "Adds duration bins, rest ratio, note density, and average duration.",
            "input_type": "checkbox",
        },
    )
    include_meter_tempo = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="includeMeterTempo",
        load_default=False,
        metadata={
            "label": "Include Meter & Tempo",
            "description": "Adds tempo statistics, change counts, and first meter flags.",
            "input_type": "checkbox",
        },
    )
    include_texture = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="includeTexture",
        load_default=False,
        metadata={
            "label": "Include Texture",
            "description": "Adds part/chord texture statistics.",
            "input_type": "checkbox",
        },
    )
    include_dynamics = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="includeDynamics",
        load_default=False,
        metadata={
            "label": "Include Dynamics",
            "description": "Adds dynamic-mark ratios and MIDI velocity statistics.",
            "input_type": "checkbox",
        },
    )
    include_tonality = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="includeTonality",
        load_default=False,
        metadata={
            "label": "Include Tonality",
            "description": "Adds key-mode and key-change features.",
            "input_type": "checkbox",
        },
    )
    include_harmony = ma.fields.Boolean(
        required=False,
        allow_none=False,
        data_key="includeHarmony",
        load_default=False,
        metadata={
            "label": "Include Harmony",
            "description": "Adds chord-root, chord-quality, and harmonic-rhythm features.",
            "input_type": "checkbox",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)


def resolve_feature_selection(
    *,
    feature_preset: str,
    include_pitch_stats: bool = False,
    include_intervals: bool = False,
    include_rhythm: bool = False,
    include_meter_tempo: bool = False,
    include_texture: bool = False,
    include_dynamics: bool = False,
    include_tonality: bool = False,
    include_harmony: bool = False,
) -> dict[str, bool]:
    normalized_preset = (feature_preset or "basic").strip().lower()
    if normalized_preset not in FEATURE_PRESETS:
        normalized_preset = "basic"

    if normalized_preset != "custom":
        enabled = set(FEATURE_PRESET_GROUPS[normalized_preset])
    else:
        enabled = {
            group
            for group, selected in {
                "pitch_stats": include_pitch_stats,
                "intervals": include_intervals,
                "rhythm": include_rhythm,
                "meter_tempo": include_meter_tempo,
                "texture": include_texture,
                "dynamics": include_dynamics,
                "tonality": include_tonality,
                "harmony": include_harmony,
            }.items()
            if selected
        }

    return {group: group in enabled for group in FEATURE_GROUP_ORDER}


def resolve_feature_selection_from_input(params: InputParameters) -> dict[str, bool]:
    return resolve_feature_selection(
        feature_preset=params.feature_preset,
        include_pitch_stats=params.include_pitch_stats,
        include_intervals=params.include_intervals,
        include_rhythm=params.include_rhythm,
        include_meter_tempo=params.include_meter_tempo,
        include_texture=params.include_texture,
        include_dynamics=params.include_dynamics,
        include_tonality=params.include_tonality,
        include_harmony=params.include_harmony,
    )


def resolve_feature_selection_from_values(
    values: Mapping[str, object],
) -> tuple[str, dict[str, bool]]:
    preset = str(values.get("featurePreset") or "basic")
    selection = resolve_feature_selection(
        feature_preset=preset,
        include_pitch_stats=_to_bool(values.get("includePitchStats")),
        include_intervals=_to_bool(values.get("includeIntervals")),
        include_rhythm=_to_bool(values.get("includeRhythm")),
        include_meter_tempo=_to_bool(values.get("includeMeterTempo")),
        include_texture=_to_bool(values.get("includeTexture")),
        include_dynamics=_to_bool(values.get("includeDynamics")),
        include_tonality=_to_bool(values.get("includeTonality")),
        include_harmony=_to_bool(values.get("includeHarmony")),
    )
    resolved_preset = preset.strip().lower()
    if resolved_preset not in FEATURE_PRESETS:
        resolved_preset = "basic"
    return resolved_preset, selection


def list_enabled_groups(selection: Mapping[str, bool]) -> list[str]:
    return [group for group in FEATURE_GROUP_ORDER if selection.get(group, False)]


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}
