import marshmallow as ma
from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from marshmallow import post_load

from .validation_functions import validate_layer_input

from .backend.neural_network import WeightInitEnum, NeuralNetworkEnum
from .backend.quantum_backends import QuantumBackends
from .backend.optimizer import OptimizerEnum

from dataclasses import dataclass


@dataclass(repr=False)
class InputParameters:
    entity_points_url: str
    clusters_url: str
    network_enum: NeuralNetworkEnum
    test_percentage: float
    device: QuantumBackends
    ibmq_token: str
    custom_backend: str
    shots: int
    optimizer: OptimizerEnum
    weight_init: WeightInitEnum
    step: float
    n_qubits: int
    N_total_iterations: int
    q_depth: int
    batch_size: int
    resolution: int
    weights_to_wiggle: int
    preprocess_layers: str = ""
    postprocess_layers: str = ""
    hidden_layers: str = ""
    use_default_dataset: bool = False
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
    use_default_dataset = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Use default dataset",
            "description": "Use internally generated dataset (no input files required).",
            "input_type": "checkbox",
        },
    )
    entity_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Data points URL",
            "description": "URL to a json file with the data points.",
            "input_type": "text",
        },
    )
    clusters_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="clusters",
        data_content_types="application/json",
        metadata={
            "label": "Labels URL",
            "description": "URL to a json file with the labels.",
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
    randomly_shuffle = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Shuffle",
            "description": "Randomly shuffle data before training.",
            "input_type": "checkbox",
        },
    )
    test_percentage = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Amount of test data",
            "description": "How much of the data is used as test data. 1 - only test data, 0 - only training data.",
            "input_type": "text",
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
        validate=validate_layer_input,
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
        validate=validate_layer_input,
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
        validate=validate_layer_input,
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
