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
from qhana_plugin_runner.api import EnumField
import marshmallow as ma
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)

from .validation_functions import validate_float_in_interval_else_int

from .backend.cluster_methods import MethodEnum
from .backend.metrics import MetricEnum
from .backend.algorithms import AlgorithmEnum


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    def __init__(
        self,
        entity_points_url: str,
        min_samples: int,
        metric_enum: None,
        minkowski_p: int,
        method_enum: None,
        xi: float,
        algorithm_enum: None,
        leaf_size: int,
        max_epsilon: float,
        epsilon: float = None,
        min_cluster_size: float = None,
        predecessor_correction: bool = False,
    ):
        self.entity_points_url = entity_points_url
        self.min_samples = min_samples
        self.max_epsilon = max_epsilon
        self.metric_enum = metric_enum
        self.minkowski_p = minkowski_p
        self.method_enum = method_enum
        self.epsilon = epsilon
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.algorithm_enum = algorithm_enum
        self.leaf_size = leaf_size
        self.predecessor_correction = predecessor_correction

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    entity_points_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/vector",
        data_content_types=["application/json"],
        metadata={
            "label": "Entity Point URL",
            "description": "URL to a json file containing the points.",
            "input_type": "text",
        },
    )
    min_samples = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Min Samples",
            "description": "The number of samples in a neighborhood for a point to be considered as a core point. Also, up and down steep regions can’t have more then min_samples consecutive non-steep points. Expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).",
            "input_type": "number",
        },
        validate=ma.validate.And(
            validate_float_in_interval_else_int(0, 1, min_inclusive=True, max_inclusive=True),
            ma.validate.Range(min=0, min_inclusive=True)
        )
    )
    max_epsilon = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Max Epsilon",
            "description": "The maximum distance between two samples for one to be considered as in the neighborhood of the other. If less than 0, then the value is set to np.inf and thus, will identify clusters across all scales; reducing max_eps will result in shorter run times.",
            "input_type": "number",
        },
    )
    metric_enum = EnumField(
        MetricEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Initialization",
            "description": "Metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.",
            "input_type": "select",
        },
    )
    minkowski_p = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Minkowski Parameter p",
            "description": "Parameter for the Minkowski metric from sklearn.metrics.pairwise_distances. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.",
            "input_type": "number",
        },
    )
    method_enum = EnumField(
        MethodEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Cluster Method",
            "description": "The extraction method used to extract clusters using the calculated reachability and ordering. Possible values are “xi” and “dbscan”.",
            "input_type": "select",
        },
    )
    epsilon = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Epsilon",
            "description": "The maximum distance between two samples for one to be considered as in the neighborhood of the other. If less than 0, it assumes the same value as max_eps. Used only when cluster_method='dbscan'.",
            "input_type": "number",
        },
    )
    xi = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Xi",
            "description": "Determines the minimum steepness on the reachability plot that constitutes a cluster boundary. For example, an upwards point in the reachability plot is defined by the ratio from one point to its successor being at most 1-xi. Used only when cluster_method='xi'.",
            "input_type": "number",
        },
        validate=ma.validate.Range(min=0, max=1, min_inclusive=True, max_inclusive=False)
    )
    predecessor_correction = ma.fields.Boolean(
        required=False,
        allow_none=True,
        metadata={
            "label": "Predecessor Correction",
            "description": "Correct clusters according to the predecessors calculated by OPTICS [R2c55e37003fe-2]. This parameter has minimal effect on most datasets. Used only when cluster_method='xi'.",
            "input_type": "checkbox",
        },
    )
    min_cluster_size = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Min Cluster Size",
            "description": "Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2). If less than 0, the value of min_samples is used instead. Used only when cluster_method='xi'.",
            "input_type": "number",
        },
    )
    algorithm_enum = EnumField(
        AlgorithmEnum,
        required=True,
        allow_none=False,
        metadata={
            "label": "Algorithm",
            "description": "Optional algorithm used to compute the nearest neighbors: ‘ball_tree’ will use BallTree ‘kd_tree’ will use KDTree ‘brute’ will use a brute-force search. ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.",
            "input_type": "select",
        },
    )
    leaf_size = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Leaf Size",
            "description": "Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.",
            "input_type": "number",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
