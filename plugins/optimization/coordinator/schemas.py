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
from dataclasses import dataclass
from typing import Optional, List

import marshmallow as ma
from marshmallow import post_load

from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass
class InternalData:
    objective_function_url: Optional[str] = None
    optimizer_url: Optional[str] = None
    dataset_url: Optional[str] = None
    optim_db_id: Optional[int] = None
    number_of_parameters: Optional[int] = None
    objective_function_calculation_url: Optional[str] = None


class InternalDataSchema(MaBaseSchema):
    objective_function_url = ma.fields.Url(required=False, allow_none=True)
    optimizer_url = ma.fields.Url(required=False, allow_none=True)
    dataset_url = ma.fields.Url(required=False, allow_none=True)
    optim_db_id = ma.fields.Integer(required=False, allow_none=True)
    number_of_parameters = ma.fields.Integer(required=False, allow_none=True)
    objective_function_calculation_url = ma.fields.URL(required=False, allow_none=True)

    @post_load
    def make_object(self, data, **kwargs):
        return InternalData(**data)


@dataclass
class OptimSelectionData:
    optimizer_url: str


class OptimSelectionSchema(FrontendFormBaseSchema):
    # FIXME: change to plugin selection when plugin selection RP has been merged
    optimizer_url = ma.fields.Url(
        required=True,
        allow_none=False,
        metadata={
            "label": "Optimizer",
            "description": "URL for an optimizer plugin",
            "input_type": "text",
        },
    )

    @post_load
    def make_object(self, data, **kwargs):
        return OptimSelectionData(**data)


@dataclass
class ObjFuncSelectionData:
    objective_function_url: str


class ObjFuncSelectionSchema(FrontendFormBaseSchema):
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

    @post_load
    def make_object(self, data, **kwargs):
        return ObjFuncSelectionData(**data)


@dataclass
class DatasetInput:
    dataset_url: str


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

    @post_load
    def make_object(self, data, **kwargs):
        return DatasetInput(**data)


@dataclass
class OutputData:
    last_objective_value: float
    optimized_parameters: List[float]


class OutputDataSchema(MaBaseSchema):
    last_objective_value = ma.fields.Float(required=True, allow_none=False)
    optimized_parameters = ma.fields.List(
        ma.fields.Float, required=True, allow_none=False
    )

    @post_load
    def make_object(self, data, **kwargs):
        return OutputData(**data)
