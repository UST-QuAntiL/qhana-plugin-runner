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
from qhana_plugin_runner.api import EnumField
import marshmallow as ma
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)

from .backend.entanglement_pattern import EntanglementPatternEnum
from .backend.optimizer import OptimizerEnum
from .backend.qiskit_backends import QiskitBackendEnum
from .backend.max_cut_clustering import MaxCutClusteringEnum


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        similarity_matrix_url: str,
        max_cut_enum: MaxCutClusteringEnum,
        num_clusters: int,
        optimizer: OptimizerEnum,
        max_trials: int,
        reps: int,
        entanglement_pattern_enum: EntanglementPatternEnum,
        backend: QiskitBackendEnum,
        shots: int,
        ibmq_custom_backend: str = "",
        ibmq_token: str = "",
    ):
        self.similarity_matrix_url = similarity_matrix_url
        self.max_cut_enum = max_cut_enum
        self.num_clusters = num_clusters
        self.optimizer = optimizer
        self.max_trials = max_trials
        self.reps = reps
        self.entanglement_pattern_enum = entanglement_pattern_enum
        self.backend = backend
        self.shots = shots
        self.ibmq_custom_backend = ibmq_custom_backend
        self.ibmq_token = ibmq_token

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    similarity_matrix_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Similarity Matrix URL",
            "description": "URL to a json file containing a similarity matrix.",
            "input_type": "text",
        },
    )
    max_cut_enum = EnumField(
        MaxCutClusteringEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Max Cut Type",
            "description": "Determines the max cut type.",
            "input_type": "select",
        },
    )
    num_clusters = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Clusters",
            "description": "If this parameter is set to x, then the number of clusters generated is 2^x.",
            "input_type": "number",
        },
    )
    optimizer = EnumField(
        OptimizerEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Optimizer",
            "description": "Local optimizer that improves the solution.",
            "input_type": "select",
        },
    )
    max_trials = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Max Trials",
            "description": "Maximum number of iterations that the optimizer takes.",
            "input_type": "number",
        },
    )
    reps = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Repetitions",
            "description": "Number of times the QAOA layers get repeated.",
            "input_type": "number",
        },
    )
    entanglement_pattern_enum = EnumField(
        EntanglementPatternEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Entanglement Pattern",
            "description": "Entanglement pattern.",
            "input_type": "select",
        },
    )
    backend = EnumField(
        QiskitBackendEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Quantum Backend",
            "description": "The quantum backend that will be used.",
            "input_type": "select",
        },
    )
    shots = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Shots",
            "description": "The amount of shots of the quantum backend",
            "input_type": "number",
        },
    )
    ibmq_custom_backend = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "IBMQ Custom Backend",
            "description": "Defines a ibmq custom backend",
            "input_type": "text",
        },
    )
    ibmq_token = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "IBMQ Token",
            "description": "IBMQ Token",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
