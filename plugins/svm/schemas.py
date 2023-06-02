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

from marshmallow import post_load
import marshmallow as ma
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)

from .backend.kernel import KernelEnum, EntanglementPatternEnum
from .backend.data_maps import DataMapsEnum
from .backend.qiskit_backends import QiskitBackends

from dataclasses import dataclass


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class InputParameters:
    train_points_url: str
    train_label_points_url: str
    test_points_url: str
    test_label_points_url: str
    train_kernel_url: str
    test_kernel_url: str
    regularization_C: float
    degree: int
    data_maps_enum: DataMapsEnum
    entanglement_pattern: EntanglementPatternEnum
    paulis: str
    reps: int
    shots: int
    backend: QiskitBackends
    ibmq_token: str
    custom_backend: str
    resolution: int
    kernel_enum: KernelEnum = None
    visualize: bool = False

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    train_kernel_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="kernel-matrix",
        data_content_types=["application/json"],
        metadata={
            "label": "Training kernel matrix URL",
            "description": "URL to a json file, containing a kernel matrix. Let X be the set of the training points, then the matrix is K(X, X).",
            "input_type": "text",
        },
    )
    train_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Training Entity points URL",
            "description": "URL to a json file with the entity points used to fit the svm.",
            "input_type": "text",
        },
    )
    train_label_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/label",
        data_content_types=["application/json"],
        metadata={
            "label": "Training Labels URL",
            "description": "URL to a json file containing the labels of the training entity points.",
            "input_type": "text",
        },
    )
    test_kernel_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="kernel-matrix",
        data_content_types=["application/json"],
        metadata={
            "label": "Test kernel matrix URL",
            "description": "URL to a json file, containing the kernel matrix. Let X be the set of the training points and T be the set of test points, then the matrix is K(X, T).",
            "input_type": "text",
        },
    )
    test_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Test Entity points URL",
            "description": "URL to a json file with the entity points that should be used for testing. These points will be labeled.",
            "input_type": "text",
        },
    )
    test_label_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/label",
        data_content_types=["application/json"],
        metadata={
            "label": "Test Labels URL",
            "description": "URL to a json file containing the labels of the test entity points. If no url is provided, then the accuracy will not be calculated.",
            "input_type": "text",
        },
    )
    kernel_enum = EnumField(
        KernelEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Kernel",
            "description": "Type of kernel to use for the SVM.",
            "input_type": "select",
        },
    )
    regularization_C = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Regularization parameter C",
            "description": "The strength of the regularization is inversely proportional to C. Must be strictly positive, the penalty is a squared l2 penalty.",
            "input_type": "number",
        },
    )
    degree = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Degree",
            "description": "Degree of the polynomial kernel function (poly). Ignored by all other kernels.",
            "input_type": "number",
        },
    )
    data_maps_enum = EnumField(
        DataMapsEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Data Map",
            "description": 'The kernels by Havlíček and Suzuki both use the same "kernel" proposed by Havlíček. '
            "The difference is that the kernel can take different so called feature maps. "
            "In the following the corresponding feature maps are listed. "
            "The sets are decided by the entanglement patterns"
            "and xi or similar notation represent the i'th entry of vector x.\n"
            "- Havlíček: f(x) = xi for set {i}, f(x) = (π - xa) * (π - xb) * ... for set {a, b, ...}\n"
            "- Suzuki 8: f(x) = xi for set {i}, f(x) = π * xa * xb * ... for set {a, b, ...}\n"
            "- Suzuki 9: f(x) = xi for set {i}, f(x) = π/2 * (1 - xa) * (1 - xb) * ... for set {a, b, ...}\n"
            "- Suzuki 10: f(x) = xi for set {i}, f(x) = π * exp(g(x)) for set {a, b, ...}, "
            "where g(x) sums up all combinations |xi - xj|^2, where i != j and i, j ∈ {a, b, ...} and takes the mean.\n"
            "- Suzuki 11: f(x) = xi for set {i}, f(x) = π / (3 * cos(xa) * cos(xb) * ...) for set {a, b, ...}\n"
            "- Suzuki 12: f(x) = xi for set {i}, f(x) = π * cos(xa) * cos(xb) * ... for set {a, b, ...}\n"
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
            "In case of 3 qubits, the patterns are as follows:\n"
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
    paulis = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Pauli Matrices",
            "description": """By default the pauli kernel only uses the Z and ZZ matrices, as described 
            in [0]. With this parameter, other pauli matrices (X, Y, Z) can be used, e.g. `Z,XX,XZ,ZYZ` is a possible 
            input.""",
            "input_type": "text",
        },
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
        QiskitBackends,
        required=True,
        allow_none=False,
        metadata={
            "label": "Backend",
            "description": "The quantum computer or simulator that will be used.",
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
    visualize = ma.fields.Boolean(
        required=False,
        allow_none=True,
        metadata={
            "label": "Visualize classification",
            "description": "Plot the decision boundary and the support vectors for the trained classifier.",
            "input_type": "checkbox",
        },
    )
    resolution = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Resolution",
            "description": "Resolution of the visualization. How finegrained the evaluation of the classifier should be.",
            "input_type": "number",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
