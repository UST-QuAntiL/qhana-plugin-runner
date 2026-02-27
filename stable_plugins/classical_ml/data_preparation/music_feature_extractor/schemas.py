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

import marshmallow as ma
from marshmallow import post_load, validate

from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema, MaBaseSchema

from .io_utils import DECLARED_FORMAT_OPTIONS, SOURCE_MODE_OPTIONS


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

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
