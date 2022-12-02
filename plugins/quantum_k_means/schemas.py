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
from marshmallow import post_load

from .backend.quantum_backend import QuantumBackends
from .backend.cluster_algos.clustering import ClusteringEnum
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        entity_points_url: str,
        clusters_cnt: int,
        variant: ClusteringEnum,
        tol: float,
        max_runs: int,
        backend: QuantumBackends,
        shots: int,
        ibmq_token: str,
        custom_backend: str,
    ):
        self.entity_points_url = entity_points_url
        self.clusters_cnt = clusters_cnt
        self.variant = variant
        self.tol = tol / 100.0
        self.max_runs = max_runs
        self.backend = backend
        self.shots = shots
        self.ibmq_token = ibmq_token
        self.custom_backend = custom_backend

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Entity points URL",
            "description": "URL to a json file with the entity points.",
            "input_type": "text",
        },
    )
    clusters_cnt = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of clusters",
            "description": "Number of clusters that shall be found.",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=1, min_inclusive=True)
    )
    variant = EnumField(
        ClusteringEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Variant",
            "description": "Variant of quantum k-means that will be used.",
            "input_type": "select",
        },
    )
    tol = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Tolerance for Convergence Criteria",
            "description": "The tolerance is given in percentage, i.e. an input of 5 = 5%.\n"
            "The algorithm does multiple iterations and after each iteration it checks how the cluster assignments for our data points "
            "have changed. If the input tolerance is 5%, then the algorithm stops, if less than 5% of the "
            "assignments have changed.",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=0, min_inclusive=True)
    )
    max_runs = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Maximum Number of Iterations",
            "description": "The algorithms does multiple iterations. After reaching the maximum number of iterations, "
            "the algorithm terminates, even if the tolerance isn't reached.",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=0, min_inclusive=True)
    )
    backend = EnumField(
        QuantumBackends,
        required=True,
        allow_none=False,
        metadata={
            "label": "Backend",
            "description": "QC or simulator that will be used.",
            "input_type": "select",
        },
    )
    shots = ma.fields.Integer(
        required=False,
        allow_none=False,
        metadata={
            "label": "Shots",
            "description": "Number of times a quantum circuit gets repeatedly executed.",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=1, min_inclusive=True)
    )
    ibmq_token = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "IBMQ Token",
            "description": "Token for IBMQ.",
            "input_type": "text",
        },
    )
    custom_backend = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "Custom backend",
            "description": "Custom backend for IBMQ.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
