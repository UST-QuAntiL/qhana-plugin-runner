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

from marshmallow import post_load, validate
import marshmallow as ma

from dataclasses import dataclass
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import FrontendFormBaseSchema, FileUrl

from .backend.quantum.pl import QNNEnum
from .backend.quantum.quantum_backends import QuantumBackends


@dataclass(repr=False)
class InputParameters:
    train_points_url: str
    test_points_url: str
    number_of_qubits: int
    embedding_size: int
    qnn_name: QNNEnum
    training_steps: int
    backend: QuantumBackends
    shots: int
    ibmq_token: str
    custom_backend: str


    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class HybridAutoencoderPennylaneRequestSchema(FrontendFormBaseSchema):
    train_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Training Entity points URL",
            "description": "URL to a json file with the entity points used to train the autoencoder.",
            "input_type": "text",
        },
    )
    test_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Test Entity points URL",
            "description": "URL to a json file with the entity points that should be used for testing. These points will be transformed by the trained autoencoder.",
            "input_type": "text",
        },
    )
    number_of_qubits = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Qubits",
            "description": "The total number of qubits used for the quantum-neural-network.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
    )
    embedding_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Embedding Size",
            "description": "The dimensionality of the embedding (number of values).",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
    )
    qnn_name = EnumField(
        QNNEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "QNN Name",
            "description": "The name of the quantum-neural-network to use as the autoencoder.",
            "input_type": "select",
        },
    )
    training_steps = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Training Steps",
            "description": "The number of training steps to train the autoencoder.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False),
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
    shots = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Shots",
            "description": "Number of shots.",
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

    @ma.validates_schema
    def validate_kernel_and_entity_points_urls(self, data, **kwargs):
        # complex errors: Depending on the case, either kernelUrl is not None or entityPointsUrl
        if data:
            number_of_qubits = data.get("number_of_qubits", None)
            embedding_size = data.get("embedding_size", None)
            if number_of_qubits is not None and embedding_size is not None:
                if embedding_size > number_of_qubits:
                    raise ma.ValidationError("The number of qubits must be greater or equal to the embedding size.")
