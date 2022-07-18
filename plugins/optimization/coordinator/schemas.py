# Copyright 2022 QHAna plugin runner contributors.
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

from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParametersSchema(FrontendFormBaseSchema):
    # FIXME: change to plugin selection when plugin selection RP has been merged
    objective_function_url = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={
            "label": "Objective Function",
            "description": "URL for an objective function plugin",
            "input_type": "text",
        },
    )


class CallbackSchema(MaBaseSchema):
    db_id = ma.fields.Integer(required=True, allow_none=False)
    number_of_parameters = ma.fields.Integer(required=True, allow_none=False)


class DatasetInputSchema(FrontendFormBaseSchema):
    dataset_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="dataset",
        data_content_types="application/json",
        metadata={
            "label": "Dataset URL",
            "description": "URL to a dataset.",
            "input_type": "text",
        },
    )
