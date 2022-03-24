import marshmallow as ma
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
)
from marshmallow import EXCLUDE, post_load


class InputParameters:
    def __init__(
        self,
        theta: float,
        step: float,
        n_qubits: int,
        N_total_iterations: int,
        q_depth: int,
        batch_size: int,
    ):
        self.theta = theta
        self.step = step
        self.n_qubits = n_qubits
        self.N_total_iterations = N_total_iterations
        self.q_depth = q_depth
        self.batch_size = batch_size


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class QNNParametersSchema(FrontendFormBaseSchema):
    theta = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Theta (Not used right now)",
            "description": "The input parameter for the QNN (rotation parameter)",
            "input_type": "text",
        },
    )
    step = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Learning Rate (0.07)",
            "description": "Learning rate for the training of the hybrid NN",
            "input_type": "text",
        },
    )
    n_qubits = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Qubits (5)",
            "description": "Number of Qubits used for the quantum circuit",
            "input_type": "text",
        },
    )
    N_total_iterations = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of iterations (2)",
            "description": "Number of total training iterations",
            "input_type": "text",
        },
    )
    q_depth = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Depth (5)",
            "description": "Depth of the quantum circuit",
            "input_type": "text",
        },
    )
    batch_size = ma.fields.Int(
        required=True,
        allow_none=False,
        metadata={
            "label": "Batch Size (10)",
            "description": "Size of the training batch",
            "input_type": "text",
        },
    )

    # ?????????????????
    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
