from xmlrpc.client import Boolean
import marshmallow as ma
from qhana_plugin_runner.api.extra_fields import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from marshmallow import EXCLUDE, post_load

from enum import Enum
import pennylane as qml
from qiskit import IBMQ

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
        print("BACKEND_ENUM", backend_enum)
        print("IBMQ_TOKEN", ibmq_token)
        print("CUSTOM_BACKEND_NAME", custom_backend_name)
        print("QUBIT_CNT", qubit_cnt)
        print("SHOTS", shots)
        if backend_enum.name.startswith("aer"):
            # Use local AER backend
            aer_backend_name = backend_enum.name[4:]
            print("AER_BACKEND_NAME", aer_backend_name)
            return qml.device(
                "qiskit.aer", wires=qubit_cnt, backend=aer_backend_name, shots=shots
            )  # TODO check if shots parameter exists
        elif backend_enum.name.startswith("ibmq"):
            # Use IBMQ backend
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=backend_enum.name,
                provider=provider,
                shots=shots,
            )  # TODO check if shots parameter exists??
        elif backend_enum.name.startswith("custom_ibmq"):
            provider = IBMQ.enable_account(ibmq_token)

            return qml.device(
                "qiskit.ibmq",
                wires=qubit_cnt,
                backend=custom_backend_name,
                provider=provider,
                shots=shots,
            )  # TODO check if shots parameter exists??
        else:
            # TASK_LOGGER.error
            print("Unknown pennylane backend specified!")  # TODO


class OptimizerEnum(Enum):
    adadelta = "Adadelta"
    adagrad = "Adagrad"
    adam = "Adam"  # TODO default
    adamW = "AdamW"
    sparse_adam = "SparseAdam"
    adamax = "Adamax"
    asgd = "ASGD"
    lbfgs = "LBFGS"
    n_adam = "NAdam"
    r_adam = "RAdam"
    rms_prob = "RMSprop"
    r_prop = "Rprop"
    sdg = "SDG"


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
        use_default_dataset=False,
        randomly_shuffle=True,
        visualize=True,
        use_quantum=True,
    ):
        self.use_default_dataset = use_default_dataset
        self.use_quantum = use_quantum
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
        self.batch_size = batch_size
        self.weight_init = weight_init
        self.randomly_shuffle = randomly_shuffle
        self.visualize = visualize


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
            "description": "Use internally generated dataset (no input files required)",
            "input_type": "checkbox",
        },
    )
    entity_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Entity points URL",
            "description": "URL to a json file with the entity points.",
            "input_type": "text",
        },
    )
    clusters_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="clusters",
        data_content_types="application/json",
        metadata={
            "label": "Clusters URL",
            "description": "URL to a json file with the clusters.",
            "input_type": "text",
        },
    )
    use_quantum = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Use quantum",
            "description": "use dressed quantum neural net instead of classical net",
            "input_type": "checkbox",
        },
    )
    visualize = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Visualize",
            "description": "Plot the decision boundary for the trained classifier",
            "input_type": "checkbox",
        },
    )
    randomly_shuffle = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Shuffle",
            "description": "Randomly shuffle data before training",
            "input_type": "checkbox",
        },
    )
    test_percentage = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Amount of test data",
            "description": "1 - only test data, 0 - only training data",
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
            "description": "Number of shots",
            "input_type": "text",
        },
    )
    optimizer = EnumField(
        OptimizerEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Optimizer",
            "description": "Type of optimizer used for training",
            "input_type": "select",
        },
    )
    step = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Learning Rate",
            "description": "Learning rate for the training of the hybrid NN",
            "input_type": "text",
        },
    )
    n_qubits = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Qubits",
            "description": "Number of Qubits used for the quantum circuit",
            "input_type": "text",
        },
    )
    N_total_iterations = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of iterations",
            "description": "Number of total training iterations",
            "input_type": "text",
        },
    )
    q_depth = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Depth",
            "description": "Depth of the quantum circuit",
            "input_type": "text",
        },
    )
    batch_size = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Batch Size",
            "description": "Size of training batches",
            "input_type": "text",
        },
    )
    weight_init = EnumField(
        WeightInitEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Weight initialization strategy",
            "description": "Distribution of (random) initialization of weigths",
            "input_type": "select",
        },
    )

    # ?????????????????
    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
