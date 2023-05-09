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
from qhana_plugin_runner.api.util import (
    FrontendFormBaseSchema,
    MaBaseSchema,
    FileUrl,
)
from celery.utils.log import get_task_logger


class TaskResponseSchema(MaBaseSchema):
    name = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_id = ma.fields.String(required=True, allow_none=False, dump_only=True)
    task_result_url = ma.fields.Url(required=True, allow_none=False, dump_only=True)


class InputParameters:
    
    def __init__(
        self,
        test_data_url: str,
        min_samples: int,
        max_epsilon: float,
        metric_enum: None,
        train_data_url: str,
        predecessor_correction: bool = False,
    ):
        self.test_data_url = test_data_url
        self.min_samples = min_samples
        self.max_epsilon = max_epsilon
        self.metric_enum = metric_enum
        self.train_data_url = train_data_url
        self.predecessor_correction = predecessor_correction

    def __str__(self):
        variables = self.__dict__.copy()
        variables["ibmq_token"] = ""
        return str(variables)


class InputParametersSchema(FrontendFormBaseSchema):
    test_data_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/shaped_vector",
        data_content_types=[
            "application/json"
        ],
        metadata={
            "label": "Test Data URL",
            "description": "URL to a json file containing the test images.",
            "input_type": "text",
        },
    )
    min_samples = ma.fields.Integer(
        required=True,
        allow_none=False,
        metadata={
            "label": "Min Samples",
            "description": "The number of samples in a neighborhood for a point to be considered as a core point. Also, up and down steep regions can’t have more then min_samples consecutive non-steep points. Expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).",
            "input_type": "number",
        },
    )
    max_epsilon = ma.fields.Float(
        required=True,
        allow_none=False,
        metadata={
            "label": "Max Epsilon",
            "description": "The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default value of np.inf will identify clusters across all scales; reducing max_eps will result in shorter run times.",
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
    train_data_url = FileUrl(
        required=True,
        allow_none=False,
        data_input_type="entity/shaped_vector",
        data_content_types=[
            "application/json"
        ],
        metadata={
            "label": "Train Data URL",
            "description": "URL to a json file containing the training images.",
            "input_type": "text",
        },
    )
    predecessor_correction = ma.fields.Boolean(
        required=True,
        allow_none=False,
        metadata={
            "label": "Predecessor Correction",
            "description": "Correct clusters according to the predecessors calculated by OPTICS [R2c55e37003fe-2]. This parameter has minimal effect on most datasets. Used only when cluster_method='xi'.",
            "input_type": "checkbox",
        },
    )

    @post_load
    def make_input_params(self, data, **kwargs) -> InputParameters:
        return InputParameters(**data)
