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
from celery.utils.log import get_task_logger
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    FileUrl,
)
from qhana_plugin_runner.util.logging import redact_log_data


TASK_LOGGER = get_task_logger(__name__)


class CircuitParameters:
    def __init__(
        self,
        circuit: str,
        executionOptions: Optional[str],
        shots: Optional[int],
        ibmqToken: Optional[str],
        backend: Optional[str],
    ):
        self.circuit = circuit
        self.executionOptions = executionOptions
        self.shots = shots
        self.ibmqToken = ibmqToken
        self.backend = backend

    def __str__(self):
        return str(redact_log_data(self.__dict__))


class AuthenticationParameters:
    def __init__(
        self,
        ibmqToken: str,
        backend: str,
    ):
        self.ibmqToken = ibmqToken
        self.backend = backend

    def __str__(self):
        return str(redact_log_data(self.__dict__))


class BackendParameters:
    def __init__(
        self,
        backend: str,
    ):
        self.backend = backend

    def __str__(self):
        return str(redact_log_data(self.__dict__))


class NoneOrInteger(ma.fields.Integer):
    """A integer field that deserializes to None if the value is an empty string. Only values >= 1 are valid."""

    def _deserialize(self, value, attr, data, **kwargs):
        if value == "":
            return None
        return super()._deserialize(value, attr, data, **kwargs)

    def _validate(self, value):
        if value is None:
            return None
        return ma.validate.Range(min=1, min_inclusive=True)(value)


class CircuitParameterSchema(FrontendFormBaseSchema):
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
    shots = NoneOrInteger(
        required=False,
        allow_none=True,
        metadata={
            "label": "Number of Shots",
            "description": """The number of times the quantum circuit gets executed. The higher, the more accurate our 
                            results get.""",
            "input_type": "number",
        },
    )
    ibmqToken = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "IBMQ Token",
            "description": "Token for IBMQ.",
            "input_type": "password",
        },
    )
    backend = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Backend",
            "description": "The quantum computer or simulator that will be used. Leave field empty to select in the next step.",
            "input_type": "text",
        },
    )

    @ma.post_load
    def make_input_params(self, data, **kwargs) -> CircuitParameters:
        return CircuitParameters(**data)


class AuthenticationParameterSchema(FrontendFormBaseSchema):
    ibmqToken = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "IBMQ Token",
            "description": "Token for IBMQ.",
            "input_type": "password",
        },
    )
    backend = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Backend",
            "description": "The quantum computer or simulator that will be used (e.g. ibmq_qasm_simulator).",
            "input_type": "text",
        },
    )

    @ma.post_load
    def make_input_params(self, data, **kwargs) -> AuthenticationParameters:
        return AuthenticationParameters(**data)


class BackendParameterSchema(FrontendFormBaseSchema):
    backend = ma.fields.String(
        required=True,
        allow_none=False,
        metadata={
            "label": "Backend",
            "description": "The quantum computer or simulator that will be used (e.g. ibmq_qasm_simulator).",
            "input_type": "text_with_datalist",
        },
    )

    @ma.post_load
    def make_input_params(self, data, **kwargs) -> BackendParameters:
        return BackendParameters(**data)
