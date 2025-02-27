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


class WebhookParams(MaBaseSchema):
    source = ma.fields.URL()
    event = ma.fields.String()


@dataclass
class OptimizerSetupTaskInputData:
    features: str
    target: str
    objective_function_plugin_selector: str
    minimizer_plugin_selector: str


class OptimizerSetupTaskInputSchema(FrontendFormBaseSchema):
    features = FileUrl(
        data_input_type="entity/vector",
        data_content_types=["text/csv", "application/json", "application/X-lines+json"],
        required=True,
        metadata={
            "label": "Features",
            "description": "A list of entities with numeric features. Each entity is 1 sample with k numeric features.",
        },
    )
    target = FileUrl(
        data_input_type="entity/vector",
        data_content_types=["text/csv", "application/json"],
        required=True,
        metadata={
            "label": "Target",
            "description": "A list of entities with the target value(s) for optimization. Each entity is 1 sample with 1 (or more) numeric target value(s).",
        },
    )
    objective_function_plugin_selector = PluginUrl(
        required=True,
        allow_none=False,
        plugin_tags=["objective-function"],
        metadata={
            "label": "Objective-Function Plugin",
            "description": "URL of objective-function-plugin.",
            "input_type": "text",
        },
    )
    minimizer_plugin_selector = PluginUrl(
        required=True,
        allow_none=False,
        plugin_tags=["minimizer"],
        metadata={
            "label": "Minimizer Plugin",
            "description": "URL of minimizer-plugin.",
            "input_type": "text",
        },
    )

    @ma.post_load
    def make_object(self, data, **kwargs):
        return OptimizerSetupTaskInputData(**data)
