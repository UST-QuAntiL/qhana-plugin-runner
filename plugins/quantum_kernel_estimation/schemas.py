from marshmallow import post_load
import marshmallow as ma
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from celery.utils.log import get_task_logger
from .backend.quantum_backends import QuantumBackends
from .backend.kernels.kernel import KernelEnum, EntanglementPatternEnum


TASK_LOGGER = get_task_logger(__name__)

class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        entity_points_url1: str,
        entity_points_url2: str,
        kernel: KernelEnum,
        entanglement_pattern: EntanglementPatternEnum,
        n_qbits: int,
        reps: int,
        shots: int,
        backend: QuantumBackends,
        ibmq_token: str,
        custom_backend: str,
    ):
        self.entity_points_url1 = entity_points_url1
        self.entity_points_url2 = entity_points_url2
        self.kernel = kernel
        self.entanglement_pattern = entanglement_pattern
        self.n_qbits = n_qbits
        self.reps = reps
        self.shots = shots
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.custom_backend = custom_backend


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url1 = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Entity points URL 1",
            "description": "URL to a json file with the entity points.",
            "input_type": "text",
        },
    )
    entity_points_url2 = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity-points",
        data_content_types="application/json",
        metadata={
            "label": "Entity points URL 2",
            "description": "URL to a json file with the entity points.",
            "input_type": "text",
        },
    )
    kernel = EnumField(
        KernelEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Quantum Kernel",
            "description": "The quantum kernel that will be used.",
            "input_type": "select",
        },
    )
    entanglement_pattern = EnumField(
        EntanglementPatternEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Entanglement Pattern",
            "description": "The entanglement pattern that will be used.",
            "input_type": "select",
        },
    )
    n_qbits = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Qubits",
            "description": "The number of qubits for the embedding that will be used.",
            "input_type": "text",
        },
    )
    reps = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Repetitions",
            "description": "The number of how often the embedding circuit gets repeated.",
            "input_type": "text",
        },
    )
    shots = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Shots",
            "description": "The number of times the quantum circuit gets executed.",
            "input_type": "text",
        },
    )
    backend = EnumField(
        QuantumBackends,
        required=True,
        allow_none=False,
        metadata={
            "label": "Backend",
            "description": "The QC or simulator that will be used.",
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

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        TASK_LOGGER.info("test")
        TASK_LOGGER.info(f"data: {data}")
        return InputParameters(**data)
