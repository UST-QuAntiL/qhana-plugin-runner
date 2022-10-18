# Copyright 2022 QHAna plugin runner contributors.
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
        if entity_points_url2 is None or entity_points_url2 == "":
            self.entity_points_url2 = entity_points_url1
        self.kernel = kernel
        self.entanglement_pattern = entanglement_pattern
        self.n_qbits = n_qbits
        self.reps = reps
        self.shots = shots
        self.backend = backend
        self.ibmq_token = ibmq_token
        self.custom_backend = custom_backend

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url1 = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Entity points URL 1",
            "description": "URL to a json file with the first entity points.",
            "input_type": "text",
        },
    )
    entity_points_url2 = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/vector",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Entity points URL 2",
            "description": "URL to a json file with the second entity points. If not provided, then 'entity points URL 1' will be used. \\"
                           "Note: To train/fit an algorithm, e.g. a Support Vector Machine (SVM), the first and second set of entity points "
                           "are equal. But to classify new points with the SVM, the first set is the training set and the second set is"
                           "the test set, i.e. the set of new points.",
            "input_type": "text",
        },
    )
    kernel = EnumField(
        KernelEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Quantum Kernel",
            "description": 'The kernels by Havlíček and Suzuki both use the same "kernel" proposed by Havlíček. '
            "The difference is that the kernel can take different so called feature maps. "
            "In the following the corresponding feature maps are listed. "
            "The sets are decided by the entanglement patterns"
            "and xi or similar notation represent the i'th entry of vector x.\n"
            "- Havlíček Kernel: f(x) = xi for set {i}, f(x) = (π - xa) * (π - xb) * ... for set {a, b, ...}\n"
            "- Suzuki Kernel 8: f(x) = xi for set {i}, f(x) = π * xa * xb * ... for set {a, b, ...}\n"
            "- Suzuki Kernel 9: f(x) = xi for set {i}, f(x) = π/2 * (1 - xa) * (1 - xb) * ... for set {a, b, ...}\n"
            "- Suzuki Kernel 10: f(x) = xi for set {i}, f(x) = π * exp(g(x)) for set {a, b, ...}, "
            "where g(x) sums up all combinations |xi - xj|^2, where i != j and i, j ∈ {a, b, ...} and takes the mean.\n"
            "- Suzuki Kernel 11: f(x) = xi for set {i}, f(x) = π / (3 * cos(xa) * cos(xb) * ...) for set {a, b, ...}\n"
            "- Suzuki Kernel 12: f(x) = xi for set {i}, f(x) = π * cos(xa) * cos(xb) * ... for set {a, b, ...}\n"
            "Suzuki actually only defined the feature maps up to a set size of 2. "
            "Here we have extended them, although it's questionable, if these are extended in a good way. "
            "For more information, please have a look at the respective papers [0] [1].",
            "input_type": "select",
        },
    )
    entanglement_pattern = EnumField(
        EntanglementPatternEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Entanglement Pattern",
            "description": "This determines how the different Qubits will be entangled."
            "In case of 3 dimensional data, we need 3 qubits and therefor the patterns are as follows:\n"
            "- full: [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}]\n"
            "- linear: [{0}, {1}, {2}, {0, 1}, {1, 2}]\n"
            "- circular: [{0}, {1}, {2}, {2, 0}, {0, 1}, {1, 2}]\n"
            "To see what this means, let's take a closer look at the case of a 'full' entanglement. The first three sets"
            "{0}, {1} and {2} in the order list, tell us that all three of these qubits will be rotated according to the"
            "feature map. Continuing, {0, 1} means both qubits 0 and 1 will be rotated by the same amount, according to"
            "the feature map and so on for {0, 2} and {1, 2}.",
            "input_type": "select",
        },
    )
    n_qbits = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Qubits",
            "description": "The number of qubits for the embedding that will be used.",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=1, min_inclusive=True),
    )
    reps = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Repetitions",
            "description": """The kernel proposed by Havlíček [0] works by creating an ansatz, which is a specific
                            quantum circuit. The final quantum circuit contains this ansatz number of repetitions times.""",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=1, min_inclusive=True),
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
