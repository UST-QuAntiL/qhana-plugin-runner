import marshmallow as ma
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from marshmallow import EXCLUDE, post_load

from enum import Enum


class OptimizerEnum(Enum):
    adam = "adam"
    sgd = "sgd"


class InputParameters:
    def __init__(
        self,
        theta: float,
        entity_points_url: str,
        clusters_url: str,
        test_percentage: float,
        step: float,
        n_qubits: int,
        N_total_iterations: int,
        q_depth: int,
        batch_size: int,
        use_default_dataset=True,
    ):
        self.use_default_dataset = use_default_dataset
        self.entity_points_url = entity_points_url
        self.clusters_url = clusters_url
        self.test_percentage = test_percentage
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
        required=False,
        allow_none=False,
        metadata={
            "label": "Theta (Not used right now)",
            "description": "The input parameter for the QNN (rotation parameter)",
            "input_type": "text",
        },
    )
    use_default_dataset = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "Use default dataset",
            "description": "Use internally generated dataset (no files required)",
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
        required=True,
        allow_none=True,
        data_input_type="clusters",
        data_content_types="application/json",
        metadata={
            "label": "Clusters URL",
            "description": "URL to a json file with the clusters.",
            "input_type": "text",
        },
    )
    test_percentage = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Test data percentage",
            "description": "Percentage of the data used for testing",
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
