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

from enum import Enum
from logging import Logger

import marshmallow as ma
from marshmallow import post_load, validate
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


class InputParameters:
    def __init__(
        self,
        entity_points_url,
        pca_type,
        dimensions,
        solver,
        batch_size,
        sparsity_alpha,
        ridge_alpha,
        kernel,
        kernel_url,
        degree,
        kernel_gamma,
        kernel_coef,
        max_itr,
        tol,
        iterated_power,
    ):
        # general parameters
        self.entity_points_url = entity_points_url
        self.pca_type = pca_type
        self.dimensions = dimensions
        # normal PCA
        self.solver = solver
        # incremental PCA
        self.batch_size = batch_size
        # sparse PCA
        self.sparsity_alpha = sparsity_alpha
        self.ridge_alpha = ridge_alpha
        # kernel PCA
        self.kernel = kernel
        self.kernel_url = kernel_url
        self.degree = degree
        self.kernel_gamma = kernel_gamma
        self.kernel_coef = kernel_coef
        self.max_itr = max_itr
        self.tol = tol
        self.iterated_power = iterated_power

    def __str__(self):
        return str(self.__dict__)


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=False,
        allow_none=True,
        data_input_type="entity/vector",
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
            "input_type": "number",
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
            "input_type": "number",
        },
    )
    sparsity_alpha = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Sparsity control",
            "description": "Sparsity controlling parameter. Higher values lead to sparser components.",
            "input_type": "number",
        },
    )
    ridge_alpha = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Ridge shrinkage",
            "description": "To avoid instability issues in case the system is under-determined, "
                           "regularization can be applied (Ridge regression) via this parameter (only for sparse PCA).",
            "input_type": "number",
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
            "input_type": "number",
        },
    )
    kernel_gamma = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Kernel Coefficient",
            "description": "Kernel coefficient 'ɣ' in rbf, poly and sigmoid kernel.",
            "input_type": "number",
        },
    )
    kernel_coef = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Independent Term in Kernel",
            "description": "Independent term 'c' in poly and sigmoid kernel.",
            "input_type": "number",
        },
    )
    max_itr = ma.fields.Integer(
        required=True,
        allow_None=False,
        metadata={
            "label": "Max Number of Iterations",
            "description": "The maximum number of iterations that sparse PCA performs.",
            "input_type": "number",
        },
        validate=validate.Range(min=0, min_inclusive=False)
    )
    tol = ma.fields.Float(
        required=True,
        allow_None=False,
        metadata={
            "label": "Error Tolerance",
            "description": "Tolerance (tol) for the stopping condition of arpack and of sparse PCA. \n"
                           "If tol <= 0, then arpack will choose the optimal value automatically and for sparse PCA, it gets set to 1e-8.",
            "input_type": "number",
        },
    )
    iterated_power = ma.fields.Integer(
        required=True,
        allow_None=False,
        metadata={
            "label": "Iterated Power",
            "description": "This sets the iterated power parameter for the randomized solver. \n"
                           "If it is set to <= 0, the iterated power will be chosen automatically.",
            "input_type": "number",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)

    @ma.validates_schema
    def validate_kernel_and_entity_points_urls(self, data, **kwargs):
        # complex errors: Depending on the case, either kernelUrl is not None or entityPointsUrl
        if data:
            if (
                data.get("kernel", None) == KernelEnum.precomputed.value
                and data.get("pca_type", None) == PCATypeEnum.kernel.value
            ):
                if data["kernel_url"] is None:
                    raise ma.ValidationError("Kernel url must not be none.")
            else:
                if data["entity_points_url"] is None:
                    raise ma.ValidationError("Entity points url must not be none.")
