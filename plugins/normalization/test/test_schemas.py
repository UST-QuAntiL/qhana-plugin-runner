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

"""Validation tests for the input schema of the normalization plugin.

The calculation task deliberately does not handle malformed user input, so
every rejection of such input has to happen in ``InputParametersSchema``.
"""

import pytest
from marshmallow import ValidationError

from .. import InputParameters, InputParametersSchema

ENTITIES_URL = "http://example.com/entities.json"
METADATA_URL = "http://example.com/attribute_metadata.json"


def _payload(**overrides) -> dict:
    """A payload as the micro frontend submits it (all fields present)."""
    payload = {
        "entitiesUrl": ENTITIES_URL,
        "attributeMetadataUrl": "",
        "attributes": "x",
        "inputRangeMin": "",
        "inputRangeMax": "",
        "outputRangeMin": 0.0,
        "outputRangeMax": 1.0,
        "useClipping": True,
        "allowMissingValues": False,
    }
    payload.update(overrides)
    return payload


def test_full_payload_is_loaded_into_input_parameters():
    params = InputParametersSchema().load(
        _payload(
            attributeMetadataUrl=METADATA_URL,
            attributes="x\ny",
            inputRangeMin=-5.0,
            inputRangeMax=5.0,
            outputRangeMin=-1.0,
            outputRangeMax=1.0,
            useClipping=False,
            allowMissingValues=True,
        )
    )

    assert isinstance(params, InputParameters)
    assert params.entities_url == ENTITIES_URL
    assert params.attribute_metadata_url == METADATA_URL
    assert params.attributes == "x\ny"
    assert params.input_range_min == -5.0
    assert params.input_range_max == 5.0
    assert params.output_range_min == -1.0
    assert params.output_range_max == 1.0
    assert params.use_clipping is False
    assert params.allow_missing_values is True


def test_empty_optional_values_are_loaded_as_none():
    params = InputParametersSchema().load(_payload())

    assert params.attribute_metadata_url is None
    assert params.input_range_min is None
    assert params.input_range_max is None


def test_flag_defaults_are_applied_when_flags_are_omitted():
    payload = _payload()
    del payload["useClipping"]
    del payload["allowMissingValues"]

    params = InputParametersSchema().load(payload)

    assert params.use_clipping is True
    assert params.allow_missing_values is False


@pytest.mark.parametrize(
    ("clipping", "expected"),
    [("true", True), ("false", False), ("1", True), ("0", False)],
)
def test_checkbox_values_are_parsed_as_booleans(clipping, expected):
    params = InputParametersSchema().load(_payload(useClipping=clipping))

    assert params.use_clipping is expected


def test_numbers_submitted_as_strings_are_parsed():
    params = InputParametersSchema().load(
        _payload(
            inputRangeMin="-2.5",
            inputRangeMax="2.5",
            outputRangeMin="0",
            outputRangeMax="100",
        )
    )

    assert params.input_range_min == -2.5
    assert params.input_range_max == 2.5
    assert params.output_range_min == 0.0
    assert params.output_range_max == 100.0


# --- Test required fields ---


def test_entities_url_is_required():
    payload = _payload()
    del payload["entitiesUrl"]

    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(payload)

    assert exc.value.messages == {"entitiesUrl": ["Missing data for required field."]}


def test_empty_entities_url_is_rejected():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(entitiesUrl=""))

    assert exc.value.messages == {"entitiesUrl": ["Field may not be null."]}


@pytest.mark.parametrize("url", ["not-a-url", "http://", "example.com/entities.json"])
def test_invalid_entities_url_is_rejected(url):
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(entitiesUrl=url))

    assert exc.value.messages == {"entitiesUrl": ["Not a valid URL."]}


def test_invalid_attribute_metadata_url_is_rejected():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(attributeMetadataUrl="not-a-url"))

    assert exc.value.messages == {"attributeMetadataUrl": ["Not a valid URL."]}


def test_attributes_are_required():
    payload = _payload()
    del payload["attributes"]

    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(payload)

    assert exc.value.messages == {"attributes": ["Missing data for required field."]}


def test_attributes_must_not_be_null():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(attributes=None))

    assert exc.value.messages == {"attributes": ["Field may not be null."]}


@pytest.mark.parametrize("field", ["outputRangeMin", "outputRangeMax"])
def test_output_range_bounds_are_required(field):
    payload = _payload()
    del payload[field]

    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(payload)

    assert exc.value.messages == {field: ["Missing data for required field."]}


def test_partially_submitted_form_without_output_range_is_rejected():
    """The micro frontend validates partial input, there the fields may be absent."""
    payload = _payload()
    del payload["outputRangeMin"]
    del payload["outputRangeMax"]

    with pytest.raises(ValidationError) as exc:
        InputParametersSchema(partial=True).load(payload)

    assert exc.value.messages == {
        "output_range_max": ["An output minimum and maximum is required."]
    }


# --- Test validation ---


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(1.0, 0.0), (0.0, 0.0), (1.0, -1.0), (-1.0, -2.0)],
)
def test_output_maximum_must_be_greater_than_minimum(minimum, maximum):
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(
            _payload(outputRangeMin=minimum, outputRangeMax=maximum)
        )

    assert exc.value.messages == {
        "output_range_max": [
            "The output maximum must be greater than the output minimum and."
        ]
    }


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(1.0, 0.0), (0.0, 0.0), (5.0, -5.0)],
)
def test_input_maximum_must_be_greater_than_minimum(minimum, maximum):
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(
            _payload(inputRangeMin=minimum, inputRangeMax=maximum)
        )

    assert exc.value.messages == {
        "input_range_max": ["The input maximum must be greater than the input minimum."]
    }


@pytest.mark.parametrize("bound", ["inputRangeMin", "inputRangeMax"])
def test_a_single_input_range_bound_is_allowed(bound):
    params = InputParametersSchema().load(_payload(**{bound: 42.0}))

    if bound == "inputRangeMin":
        assert params.input_range_min == 42.0
        assert params.input_range_max is None
    else:
        assert params.input_range_max == 42.0
        assert params.input_range_min is None


@pytest.mark.parametrize(
    "field", ["inputRangeMin", "inputRangeMax", "outputRangeMin", "outputRangeMax"]
)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_range_bounds_are_rejected(field, value):
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(**{field: value}))

    assert field in exc.value.messages


@pytest.mark.parametrize(
    "field", ["inputRangeMin", "inputRangeMax", "outputRangeMin", "outputRangeMax"]
)
def test_non_numeric_range_bounds_are_rejected(field):
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(**{field: "abc"}))

    assert exc.value.messages == {field: ["Not a valid number."]}


def test_output_range_must_not_be_null():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(outputRangeMax=None))

    assert exc.value.messages == {"outputRangeMax": ["Field may not be null."]}


# --- Test serialization ---


def test_unknown_fields_are_rejected_by_default():
    with pytest.raises(ValidationError) as exc:
        InputParametersSchema().load(_payload(unknownField="value"))

    assert exc.value.messages == {"unknownField": ["Unknown field."]}


def test_dumped_parameters_can_be_loaded_again():
    """``ProcessView`` dumps the parameters, the task loads that dump again."""
    schema = InputParametersSchema()
    params = schema.load(_payload(attributes="x\ny", inputRangeMin=0.0))

    reloaded = schema.loads(schema.dumps(params))

    assert reloaded == params
