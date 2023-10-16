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

import marshmallow as ma
from marshmallow import post_load, validate

from .backend.quantum_backends import QuantumBackends
from .backend.qknns.qknn import QkNNEnum
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)

from dataclasses import dataclass


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class InputParameters:
    train_points_url: str
    train_label_points_url: str
    test_points_url: str
    test_label_points_url: str
    variant: QkNNEnum
    k: int
    exp_itr: int
    slack: float
    backend: QuantumBackends
    shots: int
    ibmq_token: str
    custom_backend: str
    resolution: int
    minimize_qubit_count: bool = False
    visualize: bool = False

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    train_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types="application/json",
        metadata={
            "label": "Training Entity points URL",
            "description": "URL to a json file with the entity points to train the quantum kNN algorithm.",
            "input_type": "text",
        },
    )
    train_label_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/label",
        data_content_types="application/json",
        metadata={
            "label": "Training Labels URL",
            "description": "URL to a json file containing the labels of the training entity points.",
            "input_type": "text",
        },
    )
    test_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types="application/json",
        metadata={
            "label": "Test Entity points URL",
            "description": "URL to a json file with the entity points that should be used for testing. These points will be labeled.",
            "input_type": "text",
        },
    )
    test_label_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/label",
        data_content_types="application/json",
        metadata={
            "label": "Test Labels URL",
            "description": "URL to a json file containing the labels of the test entity points. If no url is provided, "
            "then the accuracy will not be calculated.",
            "input_type": "text",
        },
    )
    variant = EnumField(
        QkNNEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Variant",
            "description": "Prerequisites: \n"
            "- If a QkNN variant uses the Hamming distance, then only binary data input is allowed!\n"
            "- Datasets for schuld qknn or any QkNN that contains 'simple', may only contain a point at most once.\n"
            "QkNN descriptions:\n"
            "- schuld qknn: uses the Hamming distance as a metric and all training points as neighbours, i.e. k := Number training points\n"
            "- simple hamming qknn: uses the Hamming distance as a metric. The quantum algorithm is only used to calculate the distances. Computing the k closest neighours and doing the majority voting is then done classicaly. It is also described in [0].\n"
            "- simple fidelity qknn: analogously like simple hamming qknn, but it uses the fidelity between the data points as a similarity metric (see [1] about the fidelity metric).\n"
            "- simple angle qknn: analogously like simple hamming qknn, but it uses the angle between the data points as a distance metric.\n"
            "- basheer hamming qknn: uses the Hamming distance. It uses amplitude amplification to find the k nearest neighours as described by Basheer et al. in [1], with the oracle of Ruan et al. [2]",
            "input_type": "select",
        },
    )
    k = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Neighbours",
            "description": "The number of neighbours that the kNN algorithm will use, to label the unlabeled entity points.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
    )
    exp_itr = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Exponential Iterations",
            "description": "The maximum number of exponential search iterations for amplitude amplification.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
    )
    slack = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Slack Break Condition",
            "description": "Without any errors, this algorithm takes "
            "`max_number` many iterations. Since quantum computers aren't without errors, this input parameter `slack` "
            "allows `max_number` to be increased to `max_number = (1+slack)*max_number`.",
            "input_type": "number",
        },
    )
    minimize_qubit_count = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Minimize Qubit Count",
            "description": "If checked, then the amount of qubits used will be minimized. "
            "A consequence of this is an increased circuit depth.<br>"
            "Minimizing the qubit count is good, when the chosen quantum backend is a classical simulator.",
            "input_type": "checkbox",
        },
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
            "description": "Number of times the quantum kNN circuit gets repeated. "
            "Rule of thumb is, the higher the number of shots, the more accurate the results.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
    )
    ibmq_token = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "IBMQ Token",
            "description": "Token for IBMQ.",
            "input_type": "password",
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
    visualize = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Visualize",
            "description": "Plot the decision boundary for the trained classifier.",
            "input_type": "checkbox",
        },
    )
    resolution = ma.fields.Integer(
        required=False,
        allow_none=False,
        metadata={
            "label": "Resolution",
            "description": "The resolution of the visualization. How finegrained the evaluation of the classifier should be.\n"
            "If set to 0, only the test and training points get plotted.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=True),
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
