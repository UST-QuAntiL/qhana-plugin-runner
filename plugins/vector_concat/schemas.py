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

import marshmallow as ma

from qhana_plugin_runner.api.util import FrontendFormBaseSchema

_VALID_SUFFIX = re.compile(r"^[A-Za-z0-9._-]+$")


ACCEPTED_CONTENT_TYPES = [
    "text/csv",
    "application/json",
    "application/X-lines+json",
    "application/zip",
]


def _validate_suffix(value: str):
    if not value:
        return
    if not _VALID_SUFFIX.match(value):
        raise ma.ValidationError(
            "Suffix may only contain letters, digits, '.', '_' and '-'."
        )


def _validate_urls(value: str):
    url_validator = ma.validate.URL(schemes={"http", "https"})
    errors = []
    urls = [line.strip() for line in value.splitlines() if line.strip()]

    if not urls:
        raise ma.ValidationError("At least one URL is required.")

    for i, url in enumerate(urls, start=1):
        try:
            url_validator(url)
        except ma.ValidationError as e:
            errors.append(f"Line {i}: {e.messages[0]}")

    if errors:
        raise ma.ValidationError(errors)


class VectorConcatSchema(FrontendFormBaseSchema):
    urls = ma.fields.String(
        required=True,
        validate=_validate_urls,
        metadata={
            "label": "URLs",
            "description": "URLs of input entity/vector files",
            "input_type": "textarea",
            "accepted_content_types": ACCEPTED_CONTENT_TYPES,
        },
    )
    output_format = ma.fields.String(
        load_default="csv",
        validate=ma.validate.OneOf(("csv", "json", "lines")),
        metadata={
            "label": "Output Format",
            "description": "Format of the output data.",
            "input_type": "select",
            "options": {
                "csv": "CSV",
                "json": "JSON",
                "lines": "JSON Lines",
            },
        },
    )
    output_suffix = ma.fields.String(
        required=False,
        load_default="",
        validate=_validate_suffix,
        metadata={
            "label": "Output File Suffix",
            "description": (
                "Optional suffix for the output filename, e.g. 'something' produces 'concatenated_something.csv'."
                "Leave empty for 'concatenated.csv'."
            ),
            "input_type": "text",
        },
    )
