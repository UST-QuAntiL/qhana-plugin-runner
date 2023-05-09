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
        train_data_url: str,
        train_label_url: str,
        test_data_url: str,
        test_label_url: str,
        epochs: int,
        optimizer: None,
        lr: float,
        qcnn_enum: None,
        num_layers: int,
        batch_size: int,
        weight_init: None,
        backend: None,
        shots: int,
        ibmq_token: str,
        custom_backend: str,
        randomly_shuffle: bool = False,
        weights_to_wiggle: bool = False,
    ):
        self.train_data_url = train_data_url
        self.train_label_url = train_label_url
        self.test_data_url = test_data_url
        self.test_label_url = test_label_url
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr = lr
        self.qcnn_enum = qcnn_enum
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.backend = backend
        self.shots = shots
        self.ibmq_token = ibmq_token
        self.custom_backend = custom_backend
        self.randomly_shuffle = randomly_shuffle
        self.weights_to_wiggle = weights_to_wiggle

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    train_data_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/shaped_vector",
        data_content_types=[
            "application/json"
        ],
        metadata={
            "label": "Training Data URL",
            "description": "URL to a json file containing the training images.",
            "input_type": "text",
        },
    )
    train_label_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/label",
        data_content_types=[
            "application/json"
        ],
        metadata={
            "label": "Training Labels URL",
            "description": "URL to a json file containing the labels of the training images.",
            "input_type": "text",
        },
    )
    test_data_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/shaped_vector",
        data_content_types=[
            "application/json"
        ],
        metadata={
            "label": "Test Data URL",
            "description": "URL to a json file containing the test images. These images will be labeled.",
            "input_type": "text",
        },
    )
    test_label_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/shaped_vector",
        data_content_types=[
            "application/json"
        ],
        metadata={
            "label": "Train Data URL",
            "description": "URL to a json file containing the labels of the test entity points. If no url is provided, then the accuracy will not be calculated.",
            "input_type": "text",
        },
    )
    randomly_shuffle = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Shuffle",
            "description": "Randomly shuffle data before training.",
            "input_type": "checkbox",
        },
    )
    epochs = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Epochs",
            "description": "Number of total training epochs.",
            "input_type": "number",
        },
    )
    optimizer = EnumField(
        OptimizerEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Optimizer",
            "description": "Type of optimizer used for training.",
            "input_type": "select",
        },
    )
    lr = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Learning Rate",
            "description": "Learning rate for the training of the QCNN.",
            "input_type": "number",
        },
    )
    qcnn_enum = EnumField(
        QCNNEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Quantum Convolutional Neural Network (QCNN).",
            "description": "QCNN used to classify images.",
            "input_type": "select",
        },
    )
    num_layers = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Layers",
            "description": "The number of layers of the QCNN.",
            "input_type": "number",
        },
    )
    batch_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Batch Size",
            "description": "The size of training batches.",
            "input_type": "number",
        },
    )
    weight_init = EnumField(
        WeightInitEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Weight initialization strategy",
            "description": "Distribution of (random) initialization of weigths.",
            "input_type": "select",
        },
    )
    weights_to_wiggle = ma.fields.Boolean(
        required=True,
        allow_none=False,
        metadata={
            "label": "Quantum Layer: Weights to wiggle",
            "description": "The number of weights in the quantum circuit to update in one optimization step. 0 means all.",
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
        required=True,
        allow_none=False,
        metadata={
            "label": "Shots",
            "description": "Number of shots.",
            "input_type": "number",
        },
    )
    ibmq_token = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "IBMQ Token",
            "description": "Token for IBMQ.",
            "input_type": "text",
        },
    )
    custom_backend = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Custom backend",
            "description": "Custom backend for IBMQ.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
