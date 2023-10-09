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

from marshmallow import post_load, validate
import marshmallow as ma
from qhana_plugin_runner.api import EnumField
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from celery.utils.log import get_task_logger
from .backend.feature_map import FeatureMapEnum, EntanglementPatternEnum
from .backend.optimizer import OptimizerEnum
from .backend.vqc import AnsatzEnum
from .backend.qiskit_backends import QiskitBackends

from dataclasses import dataclass


TASK_LOGGER = get_task_logger(__name__)


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


@dataclass(repr=False)
class InputParameters:
    train_data_url: str
    train_labels_url: str
    test_data_url: str
    test_labels_url: str
    feature_map: FeatureMapEnum
    entanglement_pattern_feature_map: EntanglementPatternEnum
    reps_feature_map: int
    paulis: str
    vqc_ansatz: AnsatzEnum
    entanglement_pattern_ansatz: EntanglementPatternEnum
    reps_ansatz: int
    optimizer: OptimizerEnum
    maxitr: int
    shots: int
    backend: QiskitBackends
    ibmq_token: str
    custom_backend: str
    resolution: int

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    train_data_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Training Data URL",
            "description": "URL to a json file with the set of entity points. These entities will be used for training.",
            "input_type": "text",
        },
    )
    train_labels_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/label",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Training Label URL",
            "description": "URL to a json file with the set of labels of the training data. These labels will be used for training.",
            "input_type": "text",
        },
    )
    test_data_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Test Data URL",
            "description": "URL to a json file with the set of entity points. The plugin will predict the labels of "
            "these entity points.",
            "input_type": "text",
        },
    )
    test_labels_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/label",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Test Label URL",
            "description": "URL to a json file with the set of labels of the test data. This input is optional and will be used to calculate the accuracy.",
            "input_type": "text",
        },
    )
    feature_map = EnumField(
        FeatureMapEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Feature Map",
            "description": "Feature map",
            "input_type": "select",
        },
    )
    entanglement_pattern_feature_map = EnumField(
        EntanglementPatternEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Entanglement Pattern for the Feature Map",
            "description": "This determines how the different Qubits will be entangled."
            "In case of 3 qubits, the patterns are as follows:\n"
            "- full: [{0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}]\n"
            "- linear: [{0}, {1}, {2}, {0, 1}, {1, 2}]\n"
            "- circular: [{0}, {1}, {2}, {2, 0}, {0, 1}, {1, 2}]\n"
            "To see what this means, let's take a closer look at the case of a 'full' entanglement. The first three sets "
            "{0}, {1} and {2} in the ordered list, tell us that all three of these qubits will be rotated according to the "
            "feature map. Continuing, {0, 1} means both qubits 0 and 1 will be rotated together by the same amount, according to "
            "the feature map and so on for {0, 2} and {1, 2}.",
            "input_type": "select",
        },
    )
    reps_feature_map = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Feature Map Repetitions",
            "description": """The kernel proposed by Havlíček et al. [1] works by creating an ansatz, which is a specific
                                quantum circuit. The final quantum circuit repeats the feature map this many times.""",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=1, min_inclusive=True),
    )
    paulis = ma.fields.String(
        required=False,
        allow_none=True,
        metadata={
            "label": "Pauli Matrices",
            "description": """By default the pauli kernel only uses the Z and ZZ matrices, as described 
                    in [1]. With this parameter, other pauli matrices (X, Y, Z) can be used, e.g. `Z,XX,XZ,ZYZ` is a possible 
                    input.""",
            "input_type": "text",
        },
    )
    vqc_ansatz = EnumField(
        AnsatzEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Ansatz",
            "description": "Determines the ansatz used in the VQC.",
            "input_type": "select",
        },
    )
    entanglement_pattern_ansatz = EnumField(
        EntanglementPatternEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Entanglement Pattern for the Ansatz",
            "description": "This determines how the different Qubits will be entangled. "
            "This is the same procedure as for the feature map, except this time its for the vqc-ansatz.",
            "input_type": "select",
        },
    )
    reps_ansatz = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Number of Ansatz Repetitions",
            "description": """The chosen ansatz will be repeated this many number of repetitions.""",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=1, min_inclusive=True),
    )
    optimizer = EnumField(
        OptimizerEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Optimizer",
            "description": "This parameter determines the optimizer used to optimize the VQC's parameters. "
            "Optimizer can vary a lot, e.g. in the way they update the model's parameters or some may "
            "preserve some sort of momentum, while optimizing.",
            "input_type": "select",
        },
    )
    maxitr = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Max Number of Iterations",
            "description": """Maximum number of iterations of the optimizer""",
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
    ibmq_token = ma.fields.String(
        required=False,
        allow_none=False,
        metadata={
            "label": "IBMQ Token",
            "description": "Token for IBMQ.",
            "input_type": "password",
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
        validate=validate.Range(min=0, min_inclusive=True),
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
