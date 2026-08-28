# Copyright 2026 QHAna plugin runner contributors.
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

from types import SimpleNamespace

from router.schemas import InputParametersSchema
from router.tasks_helpers import has_enough_pca_dimensions

from tests.utils import MockResponse


def _params(**overrides):
    payload = {
        "entitiesUrl": "http://localhost/entities.csv",
        "entitiesMetadataUrl": "http://localhost/meta.json",
        "taxonomiesZipUrl": "http://localhost/taxonomies.zip",
        "includeIntermediateResultsInOutput": False,
        "rootIsPartOfHierarchy": False,
        "distanceMetric": "euclidean",
        "transformer": "linear_inverse",
        "mdsDimensions": 2,
        "metric": "metric_mds",
        "nInit": 4,
        "maxIter": 300,
        "missingDataHandling": "mean",
        "concatOutput": True,
        "outputFormat": "csv",
        "reduceDimensions": True,
        "pcaType": "normal",
        "pcaDimensions": 2,
        "solver": "auto",
        "tol": 0,
        "iteratedPower": 0,
    }
    payload.update(overrides)
    return InputParametersSchema().load(payload)


def test_has_enough_pca_dimensions_returns_true_when_vector_has_more_dimensions_than_requested():
    params = _params(reduceDimensions=True, pcaDimensions=2)
    task_data = SimpleNamespace(add_task_log_entry=lambda *args, **kwargs: None)
    vector_response = MockResponse(
        "http://localhost/vector.csv",
        "text/csv",
        text="ID,dim0,dim1,dim2\nentA,1,2,3\n",
    )

    assert has_enough_pca_dimensions(task_data, params, vector_response) is True


def test_has_enough_pca_dimensions_returns_false_when_requested_dimensions_match_vector():
    params = _params(reduceDimensions=True, pcaDimensions=3)
    task_data = SimpleNamespace(add_task_log_entry=lambda *args, **kwargs: None)
    vector_response = MockResponse(
        "http://localhost/vector.csv",
        "text/csv",
        text="ID,dim0,dim1,dim2\nentA,1,2,3\n",
    )

    assert has_enough_pca_dimensions(task_data, params, vector_response) is False
