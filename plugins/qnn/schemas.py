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
from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from marshmallow import post_load

from .validation_functions import validate_floats_seperated_by_comma

from .backend.neural_network import WeightInitEnum, NeuralNetworkEnum
from .backend.neural_network.quantum_networks import DiffMethodEnum
from .backend.quantum_backends import QuantumBackends
from .backend.optimizer import OptimizerEnum

from dataclasses import dataclass


@dataclass(repr=False)
class InputParameters:
    train_points_url: str
    train_label_points_url: str
    test_points_url: str
    test_label_points_url: str
    network_enum: NeuralNetworkEnum
    device: QuantumBackends
    ibmq_token: str
    custom_backend: str
    shots: int
    optimizer: OptimizerEnum
    weight_init: WeightInitEnum
    lr: float
    n_qubits: int
    epochs: int
    q_depth: int
    batch_size: int
    resolution: int
    weights_to_wiggle: int
    diff_method: DiffMethodEnum
    preprocess_layers: str = ""
    postprocess_layers: str = ""
    hidden_layers: str = ""
    randomly_shuffle: bool = False
    visualize: bool = False

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class QNNParametersSchema(FrontendFormBaseSchema):
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
    randomly_shuffle = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Shuffle",
            "description": "Randomly shuffle data before training.",
            "input_type": "checkbox",
        },
    )
    epochs = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Epochs",
            "description": "Number of total training epochs.",
            "input_type": "text",
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
            "description": "Learning rate for the training of the hybrid NN.",
            "input_type": "text",
        },
    )
    network_enum = EnumField(
        NeuralNetworkEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Neural Network",
            "description": "This determines the neural network used. Currently available networks are:\n"
            "- Feed Forward Neural Network: This is a simple classical feed forward neural network, "
            "with relu as an activation function after each layer.\n"
            "- Dressed quantum neural network: This is a quantum neural network with a classical feed "
            "forward neural network for preprocessing and one for postprocessing.",
            "input_type": "select",
        },
    )
    n_qubits = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of qubits",
            "description": "Number of qubits used for the quantum circuit. Or number of features per layer of the classical NN",
            "input_type": "text",
        },
    )
    q_depth = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Depth",
            "description": "Depth of the quantum circuit or classical network (number of layers)",
            "input_type": "text",
        },
    )
    preprocess_layers = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Preprocessing Layers",
            "description": "Before the data is forward into the quantum neural network, it is first preprocessed with "
            "a classical network. This determines the number of neurons in the classical preprocessing "
            "step. The i'th entry represents the number of neurons in the i'th hidden layer."
            "Please separate the layer sizes by a comma, e.g. ``4,5,10,4``",
            "input_type": "text",
        },
        validate=validate_floats_seperated_by_comma,
    )
    postprocess_layers = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Postprocessing Layers",
            "description": "Before outputting the final results, the quantum neural networks output gets postprocessed "
            " with the help of a classical neural network. This determines the number of neurons in the "
            "classical postprocessing step. The i'th entry represents the number of neurons in the i'th "
            "hidden layer.",
            "input_type": "text",
        },
        validate=validate_floats_seperated_by_comma,
    )
    hidden_layers = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Hidden Layers",
            "description": "This determines the number of neurons in the neural network. The i'th entry represents the "
            "number of neurons in the i'th hidden layer.",
            "input_type": "text",
        },
        validate=validate_floats_seperated_by_comma,
    )
    batch_size = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Batch size",
            "description": "Size of training batches.",
            "input_type": "text",
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
    weights_to_wiggle = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Quantum Layer: Weights to wiggle",
            "description": "The number of weights in the quantum circuit to update in one optimization step. 0 means all.",
            "input_type": "number",
        },
    )
    diff_method = EnumField(
        DiffMethodEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Differentiation Method",
            "description": "This parameter allows to select the method used to calculate the gradients of the quantum "
            "gates. The option `Best`, chooses the best method and even allows for normal "
            "backpropagation, if a classical simulator is used.",
            "input_type": "select",
        },
    )
    device = EnumField(
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
            "input_type": "text",
        },
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
    visualize = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Visualize",
            "description": "Plot the decision boundary for the trained classifier.",
            "input_type": "checkbox",
        },
    )
    resolution = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Resolution",
            "description": "Resolution of the visualization. How finegrained the evaluation of the classifier should be.",
            "input_type": "text",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
