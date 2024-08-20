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

from marshmallow import post_load
import marshmallow as ma
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, MaBaseSchema, FileUrl
from dataclasses import dataclass


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class InputParameters:
    entity_url: str
    clusters_url: str
    do_svm: bool = False
    do_3d: bool = False

    def __str__(self):
        return str(self.__dict__)


class InputParametersSchema(FrontendFormBaseSchema):
    entity_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Entity Point URL",
            "description": "URL to a json file containing the points.",
            "input_type": "text",
        },
    )
    clusters_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/label",
        data_content_types=["application/json"],
        metadata={
            "label": "Cluster points URL",
            "description": "URL to a json file containing the cluster points.",
            "input_type": "text",
        },
    )
    do_svm = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "SVM",
            "description": "Calculate and plot Support Vector Machine.",
            "input_type": "checkbox",
        },
    )
    do_3d = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "3D",
            "description": "Plot the Data additionally in 3D.",
            "input_type": "checkbox",
        },
    )


    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
