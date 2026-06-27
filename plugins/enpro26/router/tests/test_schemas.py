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


import pytest
from marshmallow import ValidationError
from router.schemas import InputParametersSchema, InputParameters

VALID_URL = "http://localhost:9090/experiments/1/data.csv"
VALID_ZIP = "http://localhost:9090/experiments/1/taxonomies.zip"


def _payload(**overrides) -> dict:
    base = {
        "entitiesUrl": VALID_URL,
        "entitiesMetadataUrl": VALID_URL,
        "taxonomiesZipUrl": VALID_ZIP,
        "attributes": "instrumentation\npitch",
        "routingOptions": "wu_palmer",
        "transformer": "linear_inverse",
        "aggregator": "mean",
        "missingDataHandling": "ignore",
        "dimensions": 2,
        "metric": "metric_mds",
        "nInit": 4,
        "maxIter": 300,
    }
    base.update(overrides)
    return base


def test_valid_payload_loads_successfully():
    result = InputParametersSchema().load(_payload())
    assert isinstance(result, InputParameters)
    assert result.entities_url == VALID_URL
    assert result.dimensions == 2
    assert result.routing_options.name == "wu_palmer"


def test_missing_required_fields_rejected():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load({})
    assert "entitiesUrl" in exc.value.messages
    assert "attributes" in exc.value.messages


def test_invalid_urls_rejected():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(entitiesUrl="not-a-url"))
    assert "entitiesUrl" in exc.value.messages


def test_invalid_enum_rejected():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(aggregator="invalid_aggregator"))
    assert "aggregator" in exc.value.messages
