import marshmallow as ma
from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from marshmallow import post_load

from enum import Enum
import pennylane as qml
from qiskit import IBMQ

from .validation_functions import validate_layer_input

# TODO additional pennylane devices?
#   default.qubit, default.gaussian, default.qubit.tf, default.qubit.autograd
class QuantumBackends(Enum):
    custom_ibmq = "custom_ibmq"
    aer_statevector_simulator = "aer_statevector_simulator"
    aer_qasm_simulator = "aer_qasm_simulator"
    ibmq_qasm_simulator = "ibmq_qasm_simulator"
    ibmq_santiago = "ibmq_santiago"
    ibmq_manila = "ibmq_manila"
    ibmq_bogota = "ibmq_bogota"
    ibmq_quito = "ibmq_quito"
    ibmq_belem = "ibmq_belem"
    ibmq_lima = "ibmq_lima"
    ibmq_armonk = "ibmq_armonk"

    @staticmethod
    def get_pennylane_backend(
        backend_enum: "QuantumBackends",
        ibmq_token: str,
        custom_backend_name: str,
        qubit_cnt: int,
        shots: int,
    ) -> qml.Device:
        if backend_enum.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = backend_enum.name[4:]
            return qml.device(
                "qiskit.aer", wires=qubit_cnt, backend=aer_backend_name, shots=shots
            )
        elif backend_enum.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=backend_enum.name,
                provider=provider,
                shots=shots,
            )
        elif backend_enum.name.startswith("custom_ibmq"):
            # Use custom IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=custom_backend_name,
                provider=provider,
                shots=shots,
            )
        else:
            # TASK_LOGGER.error
            raise NotImplementedError("Unknown pennylane backend specified!")


class OptimizerEnum(Enum):
    adadelta = "Adadelta"
    adagrad = "Adagrad"
    adam = "Adam"
    adamW = "AdamW"
    # sparse_adam = "SparseAdam"     # "Does not support dense gradiente, please consider Adam instead"
    adamax = "Adamax"
    asgd = "ASGD"
    # lbfgs = "LBFGS"                # "Step() missing 1 required positional argument: closure"
    n_adam = "NAdam"
    r_adam = "RAdam"
    rms_prob = "RMSprop"
    # r_prop = "Rprop"               # AttributeError('Rprop')
    # sdg = "SDG"                    # AttributeError('Rprop')


class WeightInitEnum(Enum):
    standard_normal = "standard normal"
    uniform = "uniform"
    zero = "zero"


class InputParameters:
    def __init__(
        self,
        entity_points_url: str,
        clusters_url: str,
        test_percentage: float,
        device: QuantumBackends,
        ibmq_token: str,
        custom_backend: str,
        shots: int,
        optimizer: OptimizerEnum,
        weight_init: WeightInitEnum,
        step: float,
        n_qubits: int,
        N_total_iterations: int,
        q_depth: int,
        batch_size: int,
        resolution: int,
        weights_to_wiggle: int,
        preprocess_layers: str = "",
        postprocess_layers: str = "",
        use_default_dataset=False,
        randomly_shuffle=False,
        visualize=False,
        use_quantum=False,
    ):
        self.use_default_dataset = use_default_dataset
        self.use_quantum = use_quantum
        self.resolution = resolution
        self.entity_points_url = entity_points_url
        self.clusters_url = clusters_url
        self.test_percentage = test_percentage
        self.device = device
        self.ibmq_token = ibmq_token
        self.custom_backend = custom_backend
        self.shots = shots
        self.optimizer = optimizer
        self.step = step
        self.n_qubits = n_qubits
        self.N_total_iterations = N_total_iterations
        self.q_depth = q_depth
        self.preprocess_layers = preprocess_layers
        self.postprocess_layers = postprocess_layers
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.randomly_shuffle = randomly_shuffle
        self.visualize = visualize
        self.weights_to_wiggle = weights_to_wiggle


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
    use_quantum = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Use quantum",
            "description": "Use dressed quantum neural net instead of classical net.",
            "input_type": "checkbox",
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
    shots = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Shots",
            "description": "Number of shots.",
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
    step = ma.fields.Float(
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
    N_total_iterations = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of iterations",
            "description": "Number of total training iterations. I.e. number of batches used during training",
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
            "label": "Preprocessing layers",
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
            "label": "Postprocessing layers",
            "description": "Before outputting the final results, the quantum neural networks output gets postprocessed "
            " with the help of a classical neural network. This determines the number of neurons in the "
            "classical postprocessing step. The i'th entry represents the number of neurons in the i'th "
            "hidden layer.",
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

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
