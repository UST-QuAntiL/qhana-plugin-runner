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
    def __init__(self, parameter_dict: dict, TASK_LOGGER: Logger):
        self.parameter_keys = [
            "entityPointsUrl",  # general parameters
            "pcaType",
            "dimensions",
            "minmaxScale",
            "solver",  # normal PCA
            "batchSize",  # incremental PCA
            "sparsityAlpha",  # sparse PCA
            "ridgeAlpha",
            "maxItr",
            "kernel",  # kernel PCA
        ]
        self.parameter_dict = parameter_dict
        self.parameter_dict["minmaxScale"] = self.parameter_dict.get("minmaxScale", False)

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
        self.parameter_dict["batchSize"] = max(self.parameter_dict["batchSize"], self.parameter_dict["dimensions"])

    def get(self, key):
        return self.parameter_dict[key]


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity-points",
        data_content_types=["application/json", "text/csv"],  # ["application/json", "text/csv", "application/X-lines+json"],
        metadata={
            "label": "Entity points URL",
            "description": "URL to a json file with the entity points.",
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
            "label": "Dimensions ('auto', if d <= 0)",
            "description": "Number of dimensions the output will have.",
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
    minmax_scale = ma.fields.Boolean(
        required=False,
        allow_none=False,
        metadata={
            "label": "MinMax scaling features",
            "description": "Tells, if features should be scaled to be between 0 and 1 or not",
            "input_type": "checkbox",
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
    max_itr = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Max iterations",
            "description": "Maximum number of iterations to perform.",
            "input_type": "text",
        },
    )
    kernel = EnumField(
        KernelEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Kernel",
            "description": "Type of kernel that should be used.",
            "input_type": "select",
        },
    )
