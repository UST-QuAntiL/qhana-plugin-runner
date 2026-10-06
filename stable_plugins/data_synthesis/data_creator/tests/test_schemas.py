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

"""Schema validation tests for the data-creator plugin.

``InputParametersSchema`` is a marshmallow schema that

* uses :py:class:`~qhana_plugin_runner.api.extra_fields.EnumField` to bind a
  string payload value to a member of :py:class:`DataTypeEnum`,
* uses :py:meth:`marshmallow.post_load` to convert the dict into an
  :py:class:`InputParameters` dataclass,
* marks ``dataset_type``, ``num_train_points`` and ``num_test_points`` as
  required, plus per-type required fields enforced via ``@validates_schema``
  (see ``REQUIRED_FIELDS_BY_TYPE``).

The tests here exercise each of those behaviors with valid and invalid input.
They run without a Flask app, since marshmallow schemas are pure Python.

Note: ``MaBaseSchema`` rewrites field names to ``camelCase`` for the JSON
payload, so the input keys are ``datasetType``, ``numTrainPoints`` etc., even
though the produced dataclass attributes stay ``snake_case``.
"""

import pytest
from marshmallow import ValidationError

from data_creator.backend.datasets import DataTypeEnum
from data_creator.schemas import (
    REQUIRED_FIELDS_BY_TYPE,
    InputParameters,
    InputParametersSchema,
)


def _payload(dataset_type: str, **overrides) -> dict:
    """Build a payload with sensible defaults for every shared field."""
    base = {
        "datasetType": dataset_type,
        "numTrainPoints": 10,
        "numTestPoints": 5,
    }
    base.update(overrides)
    return base


# Valid payloads, one per DataTypeEnum variant.


def test_two_spirals_valid():
    schema = InputParametersSchema()
    result = schema.load(
        _payload("two_spirals", noise=0.5, turns=1.5),
    )
    assert isinstance(result, InputParameters)
    assert result.dataset_type is DataTypeEnum.two_spirals
    assert result.num_train_points == 10
    assert result.num_test_points == 5
    assert result.noise == 0.5
    assert result.turns == 1.5
    assert result.centers is None


def test_checkerboard_valid_without_optional_fields():
    schema = InputParametersSchema()
    result = schema.load(_payload("checkerboard"))
    assert result.dataset_type is DataTypeEnum.checkerboard
    assert result.noise is None
    assert result.turns is None
    assert result.centers is None


def test_blobs_valid():
    schema = InputParametersSchema()
    result = schema.load(_payload("blobs", centers=4))
    assert result.dataset_type is DataTypeEnum.blobs
    assert result.centers == 4


def test_checkerboard_3d_valid_without_optional_fields():
    schema = InputParametersSchema()
    result = schema.load(_payload("checkerboard_3d"))
    assert result.dataset_type is DataTypeEnum.checkerboard_3d


def test_blobs_3d_valid():
    schema = InputParametersSchema()
    result = schema.load(_payload("blobs_3d", centers=2))
    assert result.dataset_type is DataTypeEnum.blobs_3d
    assert result.centers == 2


def test_optional_fields_accepted_for_type_that_does_not_require_them():
    """``checkerboard`` ignores all per-type optional fields, but the schema
    still accepts and stores them when supplied."""
    schema = InputParametersSchema()
    result = schema.load(
        _payload("checkerboard", noise=0.1, turns=2.0, centers=3),
    )
    assert result.noise == 0.1
    assert result.turns == 2.0
    assert result.centers == 3


def test_zero_test_points_valid():
    schema = InputParametersSchema()
    result = schema.load(_payload("checkerboard", numTestPoints=0))
    assert result.num_test_points == 0


# Per-type required fields.


@pytest.mark.parametrize(
    ("dataset_type", "missing_field"),
    [
        ("two_spirals", "noise"),
        ("two_spirals", "turns"),
        ("blobs", "centers"),
        ("blobs_3d", "centers"),
    ],
)
def test_missing_required_field_for_type(dataset_type, missing_field):
    schema = InputParametersSchema()
    extras = {"noise": 0.5, "turns": 1.5, "centers": 3}
    extras.pop(missing_field)
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload(dataset_type, **extras))
    assert missing_field in exc.value.messages


def test_two_spirals_missing_both_required_fields_reports_both():
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload("two_spirals"))
    assert {"noise", "turns"}.issubset(exc.value.messages)


def test_required_fields_table_covers_every_enum_member():
    assert set(REQUIRED_FIELDS_BY_TYPE.keys()) == set(DataTypeEnum)


# dataset_type validation.


def test_unknown_dataset_type_rejected():
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload("not_a_real_type"))
    assert "datasetType" in exc.value.messages


def test_dataset_type_required():
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load({"numTrainPoints": 10, "numTestPoints": 5})
    assert "datasetType" in exc.value.messages


def test_dataset_type_none_rejected():
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload(None))
    assert "datasetType" in exc.value.messages


# Shared required fields.


def test_num_train_points_required():
    schema = InputParametersSchema()
    payload = _payload("checkerboard")
    payload.pop("numTrainPoints")
    with pytest.raises(ValidationError) as exc:
        schema.load(payload)
    assert "numTrainPoints" in exc.value.messages


def test_num_test_points_required():
    schema = InputParametersSchema()
    payload = _payload("checkerboard")
    payload.pop("numTestPoints")
    with pytest.raises(ValidationError) as exc:
        schema.load(payload)
    assert "numTestPoints" in exc.value.messages


# Range validators.


@pytest.mark.parametrize("value", [0, -1, -100])
def test_num_train_points_must_be_positive(value):
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload("checkerboard", numTrainPoints=value))
    assert "numTrainPoints" in exc.value.messages


@pytest.mark.parametrize("value", [-1, -100])
def test_num_test_points_must_be_non_negative(value):
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload("checkerboard", numTestPoints=value))
    assert "numTestPoints" in exc.value.messages


@pytest.mark.parametrize("value", [-0.1, -1.0])
def test_noise_must_be_non_negative(value):
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload("two_spirals", noise=value, turns=1.5))
    assert "noise" in exc.value.messages


def test_noise_zero_allowed():
    schema = InputParametersSchema()
    result = schema.load(_payload("two_spirals", noise=0, turns=1.5))
    assert result.noise == 0


@pytest.mark.parametrize("value", [0, -0.1, -2.0])
def test_turns_must_be_strictly_positive(value):
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload("two_spirals", noise=0.5, turns=value))
    assert "turns" in exc.value.messages


@pytest.mark.parametrize("value", [0, -1])
def test_centers_must_be_at_least_one(value):
    schema = InputParametersSchema()
    with pytest.raises(ValidationError) as exc:
        schema.load(_payload("blobs", centers=value))
    assert "centers" in exc.value.messages


# Type coercion / serialisation.


def test_post_load_returns_dataclass_with_snake_case_attributes():
    schema = InputParametersSchema()
    result = schema.load(_payload("blobs", centers=2))
    assert isinstance(result, InputParameters)
    assert hasattr(result, "dataset_type")
    assert hasattr(result, "num_train_points")
    assert hasattr(result, "num_test_points")


def test_str_includes_all_fields():
    """``InputParameters.__str__`` is used in plugin logging and must surface
    every field, including the optional ones."""
    params = InputParameters(
        dataset_type=DataTypeEnum.blobs,
        num_train_points=10,
        num_test_points=5,
        centers=4,
    )
    rendered = str(params)
    for key in ("dataset_type", "num_train_points", "num_test_points", "centers"):
        assert key in rendered
