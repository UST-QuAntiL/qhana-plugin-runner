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


class ParameterHandler:
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
            "degree",
            "kernelGamma",
            "kernelCoef"
        ]
        self.parameter_dict = parameter_dict

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
        self.parameter_dict["batchSize"] = max(
            self.parameter_dict["batchSize"], self.parameter_dict["dimensions"]
        )

    def get(self, key):
        return self.parameter_dict[key]


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=True,
        allow_none=False,
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
            "description": "Batch size when executing Incremental PCA",
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
            "description": "Amount of ridge shrinkage to apply"
            " in order to improve conditioning when calling the transform method.",
            "input_type": "text",
        },
    )
    kernel = EnumField(
        KernelEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Kernel",
            "description": "Type of kernel that should be used, e.g. poly kernel \'k(x,y) = (ɣ x^T y + c)^d\'",
            "input_type": "select",
        },
    )
    degree = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Degree",
            "description": "Degree \'d\' of poly kernel.",
            "input_type": "text",
        },
    )
    kernel_gamma = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Kernel Coefficient",
            "description": f"Kernel coefficient \'ɣ\' in rbf, poly and sigmoid kernel.",
            "input_type": "text",
        },
    )
    kernel_coef = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Independent Term in Kernel",
            "description": f"Independent term \'c\' in poly and sigmoid kernel.",
            "input_type": "text",
        },
    )
