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

"""Tests for the helper functions and the calculation task of the normalization plugin."""

import csv
import json

import pytest

from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from tests.utils import MockResponse, run_plugin_task

from .. import (
    InputParameters,
    _load_entities,
    _parse_numeric_value,
    calculation_task,
    normalize_entities,
)

ENTITIES_URL = "http://example.com/entities.json"
LINES_URL = "http://example.com/entities_lines.json"
CSV_URL = "http://example.com/entities.csv"
METADATA_URL = "http://example.com/attribute_metadata.json"

MIMETYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "lines": "application/X-lines+json",
}

ENTITIES = [
    {"ID": "e1", "href": "", "x": 0.0, "y": 10.0},
    {"ID": "e2", "href": "", "x": 5.0, "y": 20.0},
    {"ID": "e3", "href": "", "x": 10.0, "y": 30.0},
]

ENTITIES_JSON = json.dumps(ENTITIES)
ENTITIES_LINES = "\n".join(json.dumps(entity) for entity in ENTITIES)
ENTITIES_CSV = "\n".join(
    ['"ID","href","x","y"']
    + ['"{ID}","",{x},{y}'.format(**entity) for entity in ENTITIES]
)


def _metadata_entity(attribute: str, type_: str = "number", **overrides) -> dict:
    metadata = {
        "ID": attribute,
        "type": type_,
        "title": "",
        "description": type_,
        "multiple": False,
        "ordered": False,
        "separator": ";",
        "refTarget": None,
        "schema": None,
    }
    metadata.update(overrides)
    return metadata


METADATA_JSON = json.dumps([_metadata_entity("x"), _metadata_entity("y")])


def _metadata(*attributes: str, **overrides) -> dict:
    return {
        attribute: AttributeMetadata.from_dict(_metadata_entity(attribute, **overrides))
        for attribute in attributes
    }


def _params(**overrides) -> InputParameters:
    params = {
        "entities_url": ENTITIES_URL,
        "attribute_metadata_url": None,
        "attributes": "x",
        "input_range_min": None,
        "input_range_max": None,
        "output_range_min": 0.0,
        "output_range_max": 1.0,
        "use_clipping": True,
        "allow_missing_values": False,
    }
    params.update(overrides)
    return InputParameters(**params)


def _task_params(**overrides) -> dict:
    """Parameters as ``ProcessView`` dumps them into the processing task."""
    params = {
        "entitiesUrl": ENTITIES_URL,
        "attributeMetadataUrl": None,
        "attributes": "x",
        "inputRangeMin": None,
        "inputRangeMax": None,
        "outputRangeMin": 0.0,
        "outputRangeMax": 1.0,
        "useClipping": True,
        "allowMissingValues": False,
    }
    params.update(overrides)
    return params


def _patch_open_url(monkeypatch, responses: dict):
    monkeypatch.setattr(
        "normalization.open_url", lambda url, *args, **kwargs: responses[url]
    )


# ---------------------------------------------------------------------------
# _parse_numeric_value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 1.0), (-2, -2.0), (0.5, 0.5), ("3", 3.0), ("-4.25", -4.25), ("1e3", 1000.0)],
)
def test_parse_numeric_value_accepts_numbers(value, expected):
    assert _parse_numeric_value("e1", "x", value) == expected


@pytest.mark.parametrize("value", [True, False])
def test_parse_numeric_value_rejects_booleans(value):
    with pytest.raises(ValueError, match="non-scalar value"):
        _parse_numeric_value("e1", "x", value)


@pytest.mark.parametrize("value", [[1], (1,), {1}, {"a": 1}])
def test_parse_numeric_value_rejects_collections(value):
    with pytest.raises(ValueError, match="non-scalar value"):
        _parse_numeric_value("e1", "x", value)


@pytest.mark.parametrize("value", ["abc", "", None, object()])
def test_parse_numeric_value_rejects_non_numeric_values(value):
    with pytest.raises(ValueError, match="non-numeric value"):
        _parse_numeric_value("e1", "x", value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), "nan"])
def test_parse_numeric_value_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="non-finite value"):
        _parse_numeric_value("e1", "x", value)


def test_parse_numeric_value_error_names_entity_and_attribute():
    with pytest.raises(ValueError, match=r"Entity 'e7'.*attribute 'height'"):
        _parse_numeric_value("e7", "height", "abc")


# ---------------------------------------------------------------------------
# normalize_entities
# ---------------------------------------------------------------------------


def test_normalize_scales_attribute_to_output_range():
    result = normalize_entities(ENTITIES, _params())

    assert [entity["x"] for entity in result] == [0.0, 0.5, 1.0]


def test_normalize_keeps_other_attributes_and_does_not_mutate_input():
    entities = [dict(entity) for entity in ENTITIES]

    result = normalize_entities(entities, _params())

    assert [entity["ID"] for entity in result] == ["e1", "e2", "e3"]
    assert [entity["y"] for entity in result] == [10.0, 20.0, 30.0]
    assert entities == ENTITIES


def test_normalize_uses_an_independent_range_per_attribute():
    result = normalize_entities(ENTITIES, _params(attributes="x\ny"))

    assert [entity["x"] for entity in result] == [0.0, 0.5, 1.0]
    assert [entity["y"] for entity in result] == [0.0, 0.5, 1.0]


def test_normalize_to_symmetric_output_range():
    params = _params(output_range_min=-1.0, output_range_max=1.0)

    result = normalize_entities(ENTITIES, params)

    assert [entity["x"] for entity in result] == [-1.0, 0.0, 1.0]


def test_normalize_to_percent_output_range():
    params = _params(output_range_min=0.0, output_range_max=100.0)

    result = normalize_entities(ENTITIES, params)

    assert [entity["x"] for entity in result] == [0.0, 50.0, 100.0]


def test_normalize_with_manual_input_range():
    params = _params(input_range_min=0.0, input_range_max=20.0)

    result = normalize_entities(ENTITIES, params)

    assert [entity["x"] for entity in result] == [0.0, 0.25, 0.5]


def test_manual_input_minimum_only_uses_data_maximum():
    params = _params(input_range_min=-10.0)

    result = normalize_entities(ENTITIES, params)

    assert [entity["x"] for entity in result] == [0.5, 0.75, 1.0]


def test_manual_input_maximum_only_uses_data_minimum():
    params = _params(input_range_max=20.0)

    result = normalize_entities(ENTITIES, params)

    assert [entity["x"] for entity in result] == [0.0, 0.25, 0.5]


def test_values_outside_a_manual_input_range_are_clipped():
    params = _params(input_range_min=2.0, input_range_max=8.0, use_clipping=True)

    result = normalize_entities(ENTITIES, params)

    assert [entity["x"] for entity in result] == [0.0, 0.5, 1.0]


def test_values_outside_a_manual_input_range_raise_without_clipping():
    params = _params(input_range_min=2.0, input_range_max=8.0, use_clipping=False)

    with pytest.raises(ValueError, match="outside the input range"):
        normalize_entities(ENTITIES, params)


def test_values_inside_a_manual_input_range_are_kept_without_clipping():
    params = _params(input_range_min=-10.0, input_range_max=20.0, use_clipping=False)

    result = normalize_entities(ENTITIES, params)

    assert [entity["x"] for entity in result] == [1 / 3, 0.5, 2 / 3]


def test_missing_values_are_normalized_to_none_when_allowed():
    entities = [dict(entity) for entity in ENTITIES]
    entities[1]["x"] = None

    result = normalize_entities(entities, _params(allow_missing_values=True))

    assert [entity["x"] for entity in result] == [0.0, None, 1.0]


def test_missing_values_are_rejected_by_default():
    entities = [dict(entity) for entity in ENTITIES]
    entities[1]["x"] = None

    with pytest.raises(ValueError, match=r"Entity 'e2' has no value for attribute 'x'"):
        normalize_entities(entities, _params())


def test_missing_attribute_is_rejected_even_if_missing_values_are_allowed():
    entities = [dict(entity) for entity in ENTITIES]
    del entities[2]["x"]

    with pytest.raises(ValueError, match=r"Entity 'e3' misses attribute 'x'"):
        normalize_entities(entities, _params(allow_missing_values=True))


def test_entities_without_id_are_rejected():
    entities = [{"x": 1.0}]

    with pytest.raises(ValueError, match="must contain an ID attribute"):
        normalize_entities(entities, _params())


def test_attribute_without_any_numeric_value_is_rejected():
    entities = [dict(entity, x=None) for entity in ENTITIES]

    with pytest.raises(ValueError, match="has no numeric values to normalize"):
        normalize_entities(entities, _params(allow_missing_values=True))


def test_non_numeric_values_are_rejected():
    entities = [dict(entity) for entity in ENTITIES]
    entities[0]["x"] = "abc"

    with pytest.raises(ValueError, match="non-numeric value"):
        normalize_entities(entities, _params())


def test_no_attributes_selected_leaves_entities_unchanged():
    result = normalize_entities(ENTITIES, _params(attributes=""))

    assert result == ENTITIES


def test_numeric_attribute_metadata_is_accepted():
    result = normalize_entities(ENTITIES, _params(), _metadata("x"))

    assert [entity["x"] for entity in result] == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("type_", ["number", "integer", "int", "float", "double"])
def test_all_numeric_metadata_types_are_accepted(type_):
    normalize_entities(ENTITIES, _params(), _metadata("x", type_=type_))


def test_non_numeric_attribute_metadata_is_rejected():
    metadata = _metadata("x", type_="string")

    with pytest.raises(ValueError, match="is not declared as numeric metadata"):
        normalize_entities(ENTITIES, _params(), metadata)


def test_attribute_without_metadata_entry_is_rejected():
    metadata = _metadata("y")

    with pytest.raises(ValueError, match=r"'x' is not declared as numeric metadata"):
        normalize_entities(ENTITIES, _params(), metadata)


def test_multi_valued_attribute_metadata_is_rejected():
    metadata = _metadata("x", multiple=True)

    with pytest.raises(ValueError, match="must contain scalar values"):
        normalize_entities(ENTITIES, _params(), metadata)


# ---------------------------------------------------------------------------
# _load_entities
# ---------------------------------------------------------------------------


def test_load_entities_without_attribute_metadata(monkeypatch):
    _patch_open_url(
        monkeypatch,
        {
            ENTITIES_URL: MockResponse(
                ENTITIES_URL, "application/json", text=ENTITIES_JSON
            )
        },
    )

    entities, metadata, mimetype = _load_entities(ENTITIES_URL)

    assert entities == ENTITIES
    assert metadata == {}
    assert mimetype == "application/json"


def test_load_entities_with_explicit_metadata_url(monkeypatch):
    _patch_open_url(
        monkeypatch,
        {
            ENTITIES_URL: MockResponse(
                ENTITIES_URL, "application/json", text=ENTITIES_JSON
            ),
            METADATA_URL: MockResponse(
                METADATA_URL, "application/json", text=METADATA_JSON
            ),
        },
    )

    entities, metadata, mimetype = _load_entities(ENTITIES_URL, METADATA_URL)

    assert entities == ENTITIES
    assert set(metadata) == {"x", "y"}
    assert metadata["x"].description == "number"
    assert mimetype == "application/json"


def test_load_entities_uses_attribute_metadata_header(monkeypatch):
    _patch_open_url(
        monkeypatch,
        {
            ENTITIES_URL: MockResponse(
                ENTITIES_URL,
                "application/json",
                text=ENTITIES_JSON,
                headers={"X-Attribute-Metadata": METADATA_URL},
            ),
            METADATA_URL: MockResponse(
                METADATA_URL, "application/json", text=METADATA_JSON
            ),
        },
    )

    _, metadata, _ = _load_entities(ENTITIES_URL)

    assert set(metadata) == {"x", "y"}


def test_explicit_metadata_url_takes_precedence_over_header(monkeypatch):
    other_url = "http://example.com/other_metadata.json"
    _patch_open_url(
        monkeypatch,
        {
            ENTITIES_URL: MockResponse(
                ENTITIES_URL,
                "application/json",
                text=ENTITIES_JSON,
                headers={"X-Attribute-Metadata": other_url},
            ),
            METADATA_URL: MockResponse(
                METADATA_URL, "application/json", text=METADATA_JSON
            ),
        },
    )

    _, metadata, _ = _load_entities(ENTITIES_URL, METADATA_URL)

    assert set(metadata) == {"x", "y"}


def test_load_entities_deserializes_csv_values_with_metadata(monkeypatch):
    _patch_open_url(
        monkeypatch,
        {
            CSV_URL: MockResponse(CSV_URL, "text/csv", text=ENTITIES_CSV),
            METADATA_URL: MockResponse(
                METADATA_URL, "application/json", text=METADATA_JSON
            ),
        },
    )

    entities, _, mimetype = _load_entities(CSV_URL, METADATA_URL)

    assert mimetype == "text/csv"
    assert [entity["x"] for entity in entities] == [0.0, 5.0, 10.0]


def test_load_entities_rejects_an_empty_entity_file(monkeypatch):
    _patch_open_url(
        monkeypatch,
        {ENTITIES_URL: MockResponse(ENTITIES_URL, "application/json", text="[]")},
    )

    with pytest.raises(ValueError, match="must not be empty"):
        _load_entities(ENTITIES_URL)


# ---------------------------------------------------------------------------
# calculation_task
# ---------------------------------------------------------------------------

TASK_RESULT = "Numeric values normalized successfully."


def _run_task(monkeypatch, responses, params):
    return run_plugin_task(
        monkeypatch,
        calculation_task,  # pyright: ignore[reportArgumentType]
        "normalization",
        responses,
        params,
        expected_result=TASK_RESULT,
    )


def _entity_responses(entities_text: str, url: str, mimetype: str) -> dict:
    return {
        url: MockResponse(
            url,
            mimetype,
            text=entities_text,
            headers={"X-Attribute-Metadata": METADATA_URL},
        ),
        METADATA_URL: MockResponse(METADATA_URL, "application/json", text=METADATA_JSON),
    }


@pytest.mark.usefixtures("celery_worker")
def test_task_normalizes_json_entities(monkeypatch):
    responses = _entity_responses(ENTITIES_JSON, ENTITIES_URL, MIMETYPES["json"])

    output = _run_task(monkeypatch, responses, _task_params(attributes="x\ny"))

    assert output.file_name == "normalized_entities.json"
    assert output.file_type == "entity/list"
    assert output.mimetype == MIMETYPES["json"]

    with open(output.file_storage_data, "r") as file_:
        entities = json.load(file_)

    assert [entity["ID"] for entity in entities] == ["e1", "e2", "e3"]
    assert [entity["x"] for entity in entities] == [0.0, 0.5, 1.0]
    assert [entity["y"] for entity in entities] == [0.0, 0.5, 1.0]


@pytest.mark.usefixtures("celery_worker")
def test_task_normalizes_line_json_entities(monkeypatch):
    responses = _entity_responses(ENTITIES_LINES, LINES_URL, MIMETYPES["lines"])

    output = _run_task(
        monkeypatch, responses, _task_params(entitiesUrl=LINES_URL, attributes="x")
    )

    assert output.mimetype == MIMETYPES["lines"]

    with open(output.file_storage_data, "r") as file_:
        entities = [json.loads(line) for line in file_ if line.strip()]

    assert [entity["x"] for entity in entities] == [0.0, 0.5, 1.0]


@pytest.mark.usefixtures("celery_worker")
def test_task_normalizes_csv_entities(monkeypatch):
    responses = _entity_responses(ENTITIES_CSV, CSV_URL, MIMETYPES["csv"])

    output = _run_task(
        monkeypatch, responses, _task_params(entitiesUrl=CSV_URL, attributes="x")
    )

    assert output.file_name == "normalized_entities.csv"
    assert output.mimetype == MIMETYPES["csv"]

    with open(output.file_storage_data, "r", newline="") as file_:
        rows = list(csv.DictReader(file_))

    assert [row["ID"] for row in rows] == ["e1", "e2", "e3"]
    assert [float(row["x"]) for row in rows] == [0.0, 0.5, 1.0]
    # unselected attributes keep their original values
    assert [float(row["y"]) for row in rows] == [10.0, 20.0, 30.0]


@pytest.mark.usefixtures("celery_worker")
def test_task_uses_an_explicit_attribute_metadata_url(monkeypatch):
    responses = {
        ENTITIES_URL: MockResponse(ENTITIES_URL, MIMETYPES["json"], text=ENTITIES_JSON),
        METADATA_URL: MockResponse(METADATA_URL, "application/json", text=METADATA_JSON),
    }

    output = _run_task(
        monkeypatch, responses, _task_params(attributeMetadataUrl=METADATA_URL)
    )

    with open(output.file_storage_data, "r") as file_:
        entities = json.load(file_)

    assert [entity["x"] for entity in entities] == [0.0, 0.5, 1.0]


@pytest.mark.usefixtures("celery_worker")
def test_task_applies_the_configured_ranges(monkeypatch):
    responses = _entity_responses(ENTITIES_JSON, ENTITIES_URL, MIMETYPES["json"])

    output = _run_task(
        monkeypatch,
        responses,
        _task_params(
            inputRangeMin=0.0,
            inputRangeMax=20.0,
            outputRangeMin=-1.0,
            outputRangeMax=1.0,
        ),
    )

    with open(output.file_storage_data, "r") as file_:
        entities = json.load(file_)

    assert [entity["x"] for entity in entities] == [-1.0, -0.5, 0.0]


@pytest.mark.usefixtures("celery_worker")
def test_task_fails_for_a_non_numeric_attribute(monkeypatch):
    responses = _entity_responses(ENTITIES_JSON, ENTITIES_URL, MIMETYPES["json"])

    with pytest.raises(ValueError, match="is not declared as numeric metadata"):
        _run_task(monkeypatch, responses, _task_params(attributes="href"))


@pytest.mark.usefixtures("celery_worker")
def test_task_fails_for_missing_values(monkeypatch):
    entities = [dict(entity) for entity in ENTITIES]
    entities[1]["x"] = None
    responses = _entity_responses(json.dumps(entities), ENTITIES_URL, MIMETYPES["json"])

    with pytest.raises(ValueError, match="has no value for attribute"):
        _run_task(monkeypatch, responses, _task_params())


@pytest.mark.usefixtures("celery_worker")
def test_task_normalizes_missing_values_to_null_when_allowed(monkeypatch):
    entities = [dict(entity) for entity in ENTITIES]
    entities[1]["x"] = None
    responses = _entity_responses(json.dumps(entities), ENTITIES_URL, MIMETYPES["json"])

    output = _run_task(monkeypatch, responses, _task_params(allowMissingValues=True))

    with open(output.file_storage_data, "r") as file_:
        result = json.load(file_)

    assert [entity["x"] for entity in result] == [0.0, None, 1.0]


@pytest.mark.usefixtures("celery_worker")
def test_task_fails_for_an_empty_entity_file(monkeypatch):
    responses = {
        ENTITIES_URL: MockResponse(ENTITIES_URL, MIMETYPES["json"], text="[]"),
    }

    with pytest.raises(ValueError, match="must not be empty"):
        _run_task(monkeypatch, responses, _task_params())
