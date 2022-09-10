from enum import Enum

import marshmallow as ma
from marshmallow import post_load

from .backend.quantum_backends import QuantumBackends
from .backend.parzen_windows.parzen_window import QParzenWindowEnum
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
    def __init__(
        self,
        train_points_url: str,
        label_points_url: str,
        test_points_url: str,
        window_size: int,
        variant: QParzenWindowEnum,
        backend: QuantumBackends,
        shots: int,
        ibmq_token: str,
        custom_backend: str,
    ):
        self.train_points_url = train_points_url
        self.label_points_url = label_points_url
        self.test_points_url = test_points_url
        self.window_size = window_size
        self.variant = variant
        self.backend = backend
        self.shots = shots
        self.ibmq_token = ibmq_token
        self.custom_backend = custom_backend


class InputParametersSchema(FrontendFormBaseSchema):
    train_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Entity points URL for Training",
            "description": "URL to a json file with the entity points to train the quantum kNN algorithm.",
            "input_type": "text",
        },
    )
    label_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="labels",
        data_content_types="application/json",
        metadata={
            "label": "Labels URL",
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
            "label": "Unlabeled Entity points URL",
            "description": "URL to a json file with the unlabeled entity points. These points will be labeled.",
            "input_type": "text",
        },
    )
    window_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Window Size",
            "description": "The size of the parzen window. If it is set to 5, then only points with a distance "
                           "of 5 or less to the test point get to vote for the test point's label",
            "input_type": "text",
        },
    )
    variant = EnumField(
        QParzenWindowEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Variant",
            "description": "Variant of quantum parzen window that will be used.",
            "input_type": "select",
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

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
