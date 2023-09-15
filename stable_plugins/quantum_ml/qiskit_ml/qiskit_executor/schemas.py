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

from typing import Optional
import marshmallow as ma
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    FileUrl,
)
from celery.utils.log import get_task_logger
from .backend.qiskit_backends import get_qiskit_backend_enum


TASK_LOGGER = get_task_logger(__name__)


class CircuitSelectionInputParameters:
    def __init__(
        self,
        circuit: str,
        executionOptions: str,
        shots: int,
        ibmqToken: str,
    ):
        self.circuit = circuit
        self.executionOptions = executionOptions
        self.shots = shots
        self.ibmqToken = ibmqToken

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmqToken"] = ""
        return str(variables)


def get_get_backend_selection_parameter_schema(ibmq_token: Optional[str] = None):
    class BackendSelectionInputParameters:
        def __init__(
            self,
            backend: get_qiskit_backend_enum(ibmq_token),
            customBackend: str,
        ):
            self.backend = backend
            self.customBackend = customBackend

    class BackendSelectionParameterSchema(FrontendFormBaseSchema):
        backend = EnumField(
            get_qiskit_backend_enum(ibmq_token),
            required=True,
            allow_none=False,
            metadata={
                "label": "Backend",
                "description": "The quantum computer or simulator that will be used.",
                "input_type": "select",
            },
        )
        customBackend = ma.fields.String(
            required=False,
            allow_none=False,
            metadata={
                "label": "Custom backend",
                "description": "Custom backend for IBMQ.",
                "input_type": "text",
            },
        )

        @ma.post_load
        def make_input_params(self, data, **kwargs) -> BackendSelectionInputParameters:
            TASK_LOGGER.info(f"data: {data}")
            return BackendSelectionInputParameters(**data)

    return BackendSelectionParameterSchema


class CircuitSelectionParameterSchema(FrontendFormBaseSchema):
    circuit = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="executable/circuit",
        data_content_types="text/x-qasm",
        metadata={
            "label": "OpenQASM Circuit",
            "description": "URL to a quantum circuit in the OpenQASM format.",
            "input_type": "text",
        },
    )
    executionOptions = FileUrl(
        required=False,
        allow_none=True,
        load_missing=None,
        data_input_type="provenance/execution-options",
        data_content_types=["text/csv", "application/json", "application/X-lines+json"],
        metadata={
            "label": "Execution Options (optional)",
            "description": "URL to a file containing execution options. (optional)",
            "input_type": "text",
        },
    )
    shots = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Shots",
            "description": """The number of times the quantum circuit gets executed. The higher, the more accurate our 
                            results get.""",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=1, min_inclusive=True),
    )
    ibmqToken = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "IBMQ Token",
            "description": "Token for IBMQ.",
            "input_type": "text",
        },
    )

    @ma.post_load
    def make_input_params(self, data, **kwargs) -> CircuitSelectionInputParameters:
        log_data = data.copy()
        if "ibmqToken" in log_data:
            log_data["ibmqToken"] = "****"
        TASK_LOGGER.info(f"data: {log_data}")
        return CircuitSelectionInputParameters(**data)
