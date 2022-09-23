# Copyright 2021 QHAna plugin runner contributors.
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

from enum import Enum
from logging import Logger

import marshmallow as ma
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


class SolverEnum(Enum):
    auto = "auto"
    full = "full"
    arpack = "arpack"
    randomized = "randomized"


class PCATypeEnum(Enum):
    normal = "normal"
    incremental = "incremental"
    sparse = "sparse"
    kernel = "kernel"


class KernelEnum(Enum):
    linear = "linear"
    poly = "poly"
    rbf = "rbf"
    sigmoid = "sigmoid"
    cosine = "cosine"
    precomputed = "precomputed"


class ParameterHandler:
    """
    This class takes all the parameters set in the front end, prepares them and makes them available via a get method.
    """

    def __init__(self, parameter_dict: dict, TASK_LOGGER: Logger):
        self.parameter_keys = [
            "entityPointsUrl",  # general parameters
            "pcaType",
            "dimensions",
            "solver",  # normal PCA
            "batchSize",  # incremental PCA
            "sparsityAlpha",  # sparse PCA
            "ridgeAlpha",
            "kernel",  # kernel PCA
            "kernelUrl",
            "degree",
            "kernelGamma",
            "kernelCoef",
            "maxItr",
            "tol",
            "iteratedPower",
        ]
        self.parameter_dict = parameter_dict
        # Prevents the log and check to throw an error, when there is no entity URL,
        # if we are using the precomputed kernel option
        if (
            self.parameter_dict.get("kernel", None) == KernelEnum.precomputed.value
            and self.parameter_dict.get("pcaType", None) != PCATypeEnum.kernel
        ):
            self.parameter_dict["entityPointsUrl"] = "No URL"
        # Prevents the log and check to throw an error, when there is no kernel URL,
        # if we are NOT using the precomputed kernel option
        else:
            self.parameter_dict["kernelUrl"] = "No URL"

        # log and check input parameters
        not_provided_params = []
        for key in self.parameter_keys:
            parameter = self.parameter_dict.get(key, None)
            TASK_LOGGER.info(f"Loaded input parameters from db: {key}='{parameter}'")
            if parameter is None:
                not_provided_params.append(key)
        if len(not_provided_params) != 0:
            raise ValueError(
                f"The following inputs were not provided: {str(not_provided_params)[1:-1]}"
            )

        # Maximum number of iterations for sparse PCA should be non-negativ
        if self.parameter_dict["maxItr"] <= 0:
            raise ValueError(
                f"The maximum number of iterations should be greater than 0, but was: {str(self.parameter_dict['maxItr'])}"
            )

        # Set parameters correctly
        self.set_parameters_correctly()

    # Sets parameters to correct values
    # E.g. if dimensions <= 0, we want it to be chosen automatically => set it to None or 'mle'
    def set_parameters_correctly(self):
        # Set parameters to correct conditions
        # batch size needs to be at least the size of the dimensions
        self.parameter_dict["batchSize"] = max(
            self.parameter_dict["batchSize"], self.parameter_dict["dimensions"]
        )
        # If dimensions <= 0, then dimensions should be chosen automatically
        if self.parameter_dict["dimensions"] <= 0:
            self.parameter_dict["dimensions"] = None
            if self.parameter_dict["pcaType"] == PCATypeEnum.normal:
                self.parameter_dict["dimensions"] = "mle"
        # If tolerance tol is set to <= 0, then we set it as follows
        if self.parameter_dict["tol"] <= 0:
            # 1e-8 for sparse PCA
            if self.parameter_dict["pcaType"] == PCATypeEnum.sparse:
                self.parameter_dict["tol"] = 1e-8
            # 0 for normal and kernel PCA
            else:
                self.parameter_dict["tol"] = 0
            # Incremental PCA does not use this parameter

        # If iterated power is set to <= 0, then it should be chosen automatically
        if self.parameter_dict["iteratedPower"] <= 0:
            self.parameter_dict["iteratedPower"] = "auto"

    def get(self, key):
        return self.parameter_dict[key]


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity-points",
        data_content_types=[
            "application/json",
            "text/csv",
        ],
        metadata={
            "label": "Entity points URL",
            "description": "URL to a json/csv file with the entity points.",
            "input_type": "text",
        },
    )
    pca_type = EnumField(
        PCATypeEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "PCA Type",
            "description": "Type of PCA that will be executed.",
            "input_type": "select",
        },
    )
    dimensions = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Dimensions",
            "description": "Number of dimensions k that the output will have."
            "\nFor k <= 0, normal PCA will guess k and all other PCA types will take max k.",
            "input_type": "text",
        },
    )
    solver = EnumField(
        SolverEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Solver",
            "description": "Type of PCA solver that will be used.",
            "input_type": "select",
        },
    )
    batch_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Batch Size",
            "description": "Batch size used when executing Incremental PCA. "
            "The batch size will be automatically set to at least the number of dimensions k.",
            "input_type": "text",
        },
    )
    sparsity_alpha = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Sparsity control",
            "description": "Sparsity controlling parameter. Higher values lead to sparser components.",
            "input_type": "text",
        },
    )
    ridge_alpha = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Ridge shrinkage",
            "description": "To avoid instability issues in case the system is under-determined, "
            "regularization can be applied (Ridge regression) via this parameter (only for sparse PCA).",
            "input_type": "text",
        },
    )
    kernel = EnumField(
        KernelEnum,
        required=False,
        allow_none=True,
        metadata={
            "label": "Kernel",
            "description": "Type of kernel that should be used, e.g. poly kernel 'k(x,y) = (ɣ x^T y + c)^d'",
            "input_type": "select",
        },
    )
    kernel_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="kernel-matrix",
        data_content_types=["application/json"],
        metadata={
            "label": "Kernel matrix URL",
            "description": "URL to a json file, containing the kernel matrix."
            "Note that only kernel matrices between the same set of points X can be processed here, "
            "i.e. K(X, X)",
            "input_type": "text",
        },
    )
    degree = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Degree",
            "description": "Degree 'd' of poly kernel.",
            "input_type": "text",
        },
    )
    kernel_gamma = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Kernel Coefficient",
            "description": "Kernel coefficient 'ɣ' in rbf, poly and sigmoid kernel.",
            "input_type": "text",
        },
    )
    kernel_coef = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Independent Term in Kernel",
            "description": "Independent term 'c' in poly and sigmoid kernel.",
            "input_type": "text",
        },
    )
    max_itr = ma.fields.Integer(
        required=True,
        allow_None=False,
        metadata={
            "label": "Max Number of Iterations",
            "description": "The maximum number of iterations that sparse PCA performs.",
            "input_type": "text",
        },
    )
    tol = ma.fields.Float(
        required=True,
        allow_None=False,
        metadata={
            "label": "Error Tolerance",
            "description": "Tolerance (tol) for the stopping condition of arpack and of sparse PCA. \n"
            "If tol <= 0, then arpack will choose the optimal value automatically and for sparse PCA, it gets set to 1e-8.",
            "input_type": "text",
        },
    )
    iterated_power = ma.fields.Integer(
        required=True,
        allow_None=False,
        metadata={
            "label": "Iterated Power",
            "description": "This sets the iterated power parameter for the randomized solver. \n"
            "If it is set to <= 0, the iterated power will be chosen automatically.",
            "input_type": "text",
        },
    )
