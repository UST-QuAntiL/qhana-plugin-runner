# Copyright 2025 QHAna plugin runner contributors.
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

import marshmallow as ma

from qhana_plugin_runner.api.util import FrontendFormBaseSchema


class SQLInputSchema(FrontendFormBaseSchema):
    sql = ma.fields.String(
        required=True,
        metadata={
            "label": "SQL Command",
            "description": "DuckDB SQL query to execute.",
            "input_type": "textarea",
        },
    )
    output_format = ma.fields.String(
        missing="csv",
        validate=ma.validate.OneOf(("csv", "json")),
        metadata={
            "label": "Output Format",
            "description": "Format of the output data.",
            "input_type": "select",
            "options": {"csv": "CSV", "json": "JSON"},
        },
    )
