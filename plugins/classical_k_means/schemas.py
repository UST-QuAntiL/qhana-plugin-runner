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
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from celery.utils.log import get_task_logger


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        entity_points_url: str,
        num_clusters: int,
        maxiter: int,
        relative_residual: float,
        visualize: bool = False,
    ):
        self.entity_points_url = entity_points_url
        self.num_clusters = num_clusters
        self.maxiter = maxiter
        self.relative_residual = relative_residual
        self.visualize = visualize

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
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
    num_clusters = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Clusters",
            "description": "Determines the number of clusters generated.",
            "input_type": "number",
        },
    )
    maxiter = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Max Iterations",
            "description": "The number of k means iterations.",
            "input_type": "number",
        },
    )
    relative_residual = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Relative Residual",
            "description": "The amount in percentage of how many data points can change their labels between two runs, e.g. if set to 5, then less than 5% of the data points may change for the algorithm to be considered as converged.",
            "input_type": "number",
        },
    )
    visualize = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Visualize",
            "description": "Plot the clustered data.",
            "input_type": "checkbox",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
