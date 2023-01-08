from enum import Enum

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


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)

    def __init__(
        self,
        train_points_url: str,
        train_label_points_url: str,
        test_points_url: str,
        test_label_points_url: str,
        k: int,
        variant: QkNNEnum,
        exp_itr: int,
        backend: QuantumBackends,
        shots: int,
        ibmq_token: str,
        custom_backend: str,
        resolution: int,
        minimize_qubit_count=False,
    ):
        self.train_points_url = train_points_url
        self.train_label_points_url = train_label_points_url
        self.test_points_url = test_points_url
        self.test_label_points_url = test_label_points_url
        self.k = k
        self.variant = variant
        self.exp_itr = exp_itr
        self.minimize_qubit_count = minimize_qubit_count
        self.backend = backend
        self.shots = shots
        self.ibmq_token = ibmq_token
        self.custom_backend = custom_backend
        self.resolution = resolution


class InputParametersSchema(FrontendFormBaseSchema):
    train_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity-points",
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
        data_input_type="labels",
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
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Testing Entity points URL",
            "description": "URL to a json file with the entity points that should be used for testing. These points will be labeled.",
            "input_type": "text",
        },
    )
    test_label_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="labels",
        data_content_types="application/json",
        metadata={
            "label": "Testing Labels URL",
            "description": "URL to a json file containing the labels of the testing entity points. If no url is provided,"
                           "then the accuracy will not be calculated.",
            "input_type": "text",
        },
    )
    k = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Neighbours",
            "description": "The number of neighbours that the kNN algorithm will use, to label the unlabeled entity points.",
            "input_type": "text",
        },
    )
    variant = EnumField(
        QkNNEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Variant",
            "description": "- schuld qknn: uses the Hamming distance as a metric and all training points as neighbours, i.e. k := Number training points\n"
                           "- simple hamming qknn: uses the Hamming distance as a meric. The quantum algorithm is only used to calculate the distances. Computing the k closest neighours and doing the majority voting is then done classicaly. It is also described in [0].\n"
                           "- simple fidelity qknn: analogously like simple hamming qknn, but it uses the fidelity as a similarity metric (see [1] about the fidelity metric).\n"
                           "- basheer hamming qknn: uses the Hamming distance. It uses amplitude amplification to find the k nearest neighours as described by Basheer et al. in [1], with the oracle of Ruan et al. [2]",
            "input_type": "select",
        },
    )
    exp_itr = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Exponential Iterations",
            "description": "The maximum number of exponential search iterations for amplitude amplification.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False)
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
        }
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
            "input_type": "text",
        }
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
    resolution = ma.fields.Integer(
        required=False,
        allow_none=False,
        metadata={
            "label": "Resolution",
            "description": "The resolution of the visualization. How finegrained the evaluation of the classifier should be.\n"
                           "If set to 0, only the test and training points get plotted.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False)
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
