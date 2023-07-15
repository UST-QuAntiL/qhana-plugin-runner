# Copyright 2023 QHAna plugin runner contributors.
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

from qhana_plugin_runner.api.util import (
    FileUrl,
    FrontendFormBaseSchema,
    MaBaseSchema,
    PluginUrl,
)


class OptimizerTaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class OptimizerCallbackTaskInputSchema(FrontendFormBaseSchema):
    input_str = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Input String",
            "description": "A simple string input.",
            "input_type": "textarea",
        },
    )


@dataclass
class OptimizerSetupTaskInputData:
    input_file_url: str
    target_variable: str
    objective_function_plugin_selector: str
    minimizer_plugin_selector: str


class OptimizerSetupTaskInputSchema(FrontendFormBaseSchema):
    input_file_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="*",
        data_content_types=["text/csv"],
        metadata={
            "label": "Dataset URL",
            "description": "URL to a csv file with optimizable data.",
        },
    )
    target_variable = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Target Variable",
            "description": "Name of the target variable in the dataset.",
            "input_type": "text",
        },
    )
    minimizer_plugin_selector = PluginUrl(
        required=True,
        allow_none=False,
        plugin_tags=["minimization"],
        metadata={
            "label": "Minimizer Plugin Selector",
            "description": "URL of minimizer-plugin.",
            "input_type": "text",
        },
    )
    objective_function_plugin_selector = PluginUrl(
        required=True,
        allow_none=False,
        plugin_tags=["objective-function"],
        metadata={
            "label": "Objective-Function Plugin Selector",
            "description": "URL of objective-function-plugin.",
            "input_type": "text",
        },
    )

    @ma.post_load
    def make_object(self, data, **kwargs):
        return OptimizerSetupTaskInputData(**data)
