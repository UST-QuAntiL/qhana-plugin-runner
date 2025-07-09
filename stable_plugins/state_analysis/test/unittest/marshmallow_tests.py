import marshmallow as ma
import pytest
from common.plugin_utils.marshmallow_util import (
    ComplexNumberField,
    ComplexVectorField,
    QasmInput,
    QasmInputList,
    SetOfComplexVectorsField,
    SetOfTwoComplexVectorsField,
    ToleranceField,
)
from qhana_plugin_runner.api.util import FileUrl, FrontendFormBaseSchema


# ---------------------------
# Tolerance Field Tests
# ---------------------------
@pytest.fixture
def tolerance_schema():
    class Schema(FrontendFormBaseSchema):
        tolerance = ToleranceField()

    return Schema()


import marshmallow as ma
import pytest


def test_tolerance_valid(tolerance_schema):
    """Test that a valid tolerance value is correctly processed."""
    input_data = {"tolerance": 0.01}
    expected_output = {"tolerance": 0.01}
    result = tolerance_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_tolerance_empty_string(tolerance_schema):
    """Test that an empty string for tolerance remains unchanged."""
    input_data = {"tolerance": ""}
    expected_output = {"tolerance": ""}
    result = tolerance_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_tolerance_invalid_type(tolerance_schema):
    """Test that an invalid tolerance input raises a validation error."""
    input_data = {"tolerance": "invalid"}
    with pytest.raises(ma.ValidationError) as excinfo:
        tolerance_schema.load(input_data)
    assert "Not a valid number." in str(excinfo.value)


def test_tolerance_missing_field(tolerance_schema):
    """Test that a missing tolerance field returns an empty dictionary."""
    input_data = {}
    expected_output = {}
    result = tolerance_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_tolerance_negative_value(tolerance_schema):
    """Test that a negative tolerance value raises a validation error."""
    input_data = {"tolerance": -0.01}
    with pytest.raises(ma.ValidationError) as excinfo:
        tolerance_schema.load(input_data)
    assert "Tolerance cannot be smaller than 0" in str(excinfo.value)


# ---------------------------
# ComplexNumber Field Tests
# ---------------------------
@pytest.fixture
def complexnumber_schema():
    class Schema(FrontendFormBaseSchema):
        complexNumber = ComplexNumberField()

    return Schema()


def test_complexnumber_field_valid(complexnumber_schema):
    """Test that a valid complex number is correctly processed."""
    input_data = {"complexNumber": [2.0, 1.0]}
    expected_output = {"complexNumber": [2.0, 1.0]}
    result = complexnumber_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_complexnumber_field_string_input(complexnumber_schema):
    """Test that a complex number provided as a JSON string is correctly processed."""
    input_data = {"complexNumber": "[2.0, 1.0]"}
    expected_output = {"complexNumber": [2.0, 1.0]}
    result = complexnumber_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_complexnumber_field_invalid_structure(complexnumber_schema):
    """Test that a complex number with missing parts raises an error."""
    input_data = {"complexNumber": [2.0]}  # Missing imaginary part
    with pytest.raises(ma.ValidationError) as excinfo:
        complexnumber_schema.load(input_data)
    assert "Invalid input length. Expected two elements" in str(excinfo.value)


def test_complexnumber_field_invalid_structure_too_many(complexnumber_schema):
    """Test that a complex number with too many elements raises an error."""
    input_data = {"complexNumber": [2.0, 2.0, 2.0]}
    with pytest.raises(ma.ValidationError) as excinfo:
        complexnumber_schema.load(input_data)
    assert "Invalid input length. Expected two elements" in str(excinfo.value)


def test_complexnumber_field_invalid_types(complexnumber_schema):
    """Test that invalid types for real/imaginary parts raise an error."""
    input_data = {"complexNumber": ["real", 1.0]}  # Invalid real part
    with pytest.raises(ma.ValidationError) as excinfo:
        complexnumber_schema.load(input_data)
    assert (
        "Invalid numbers provided. Real and imaginary parts must be valid floats"
        in str(excinfo.value)
    )


def test_complexnumber_field_invalid_empty(complexnumber_schema):
    """Test that an empty string for complex number field raises an error."""
    input_data = {"complexNumber": ""}
    with pytest.raises(ma.ValidationError) as excinfo:
        complexnumber_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_complexnumber_field_invalid_non_json(complexnumber_schema):
    """Test that a non-JSON input raises an error."""
    input_data = {"complexNumber": "a1 2!"}
    with pytest.raises(ma.ValidationError) as excinfo:
        complexnumber_schema.load(input_data)
    assert "Invalid input. Expected a JSON" in str(excinfo.value)


def test_complexnumber_field_invalid_not_list_or_tuple(complexnumber_schema):
    """Test that an input not representing a list raises an error."""
    input_data = {"complexNumber": "1"}
    with pytest.raises(ma.ValidationError) as excinfo:
        complexnumber_schema.load(input_data)
    assert "Invalid input. Expected a list" in str(excinfo.value)


# ---------------------------
# ComplexVector Field Tests
# ---------------------------
@pytest.fixture
def complexvector_schema():
    class Schema(FrontendFormBaseSchema):
        complexVector = ComplexVectorField()

    return Schema()


def test_complexvector_field_valid(complexvector_schema):
    """Test that a valid complex vector is correctly processed."""
    input_data = {"complexVector": [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]]}
    expected_output = {"complexVector": [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]]}
    result = complexvector_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_complexvector_field_valid_as_string(complexvector_schema):
    """Test that a valid complex vector provided as a JSON string is correctly processed."""
    input_data = {"complexVector": "[[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]]"}
    expected_output = {"complexVector": [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]]}
    result = complexvector_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_complexvector_field_invalid_string(complexvector_schema):
    """Test that an invalid string input for complex vector raises an error."""
    input_data = {"complexVector": "1"}
    with pytest.raises(ma.ValidationError) as excinfo:
        complexvector_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_complexvector_field_invalid_empty(complexvector_schema):
    """Test that an empty string for complex vector raises an error."""
    input_data = {"complexVector": ""}
    expected_output = {"complexVector": ""}
    result = complexvector_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_complexvector_field_invalid_element(complexvector_schema):
    """Test that an invalid element within a complex vector raises an error."""
    input_data = {"complexVector": [[1.0, -2.0], "invalid_element"]}
    with pytest.raises(ma.ValidationError) as excinfo:
        complexvector_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_complexvector_field_invalid_element_as_string(complexvector_schema):
    """Test that an invalid element provided as a JSON string in a complex vector raises an error."""
    input_data = '{"invalid_element"}'
    with pytest.raises(ma.ValidationError) as excinfo:
        complexvector_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_complexvector_field_invalid_element_not_a_list(complexvector_schema):
    """Test that an element that is not a list raises an error in a complex vector."""
    input_data = {"complexVector": "invalid_element"}
    with pytest.raises(ma.ValidationError) as excinfo:
        complexvector_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_complexvector_field_empty_list(complexvector_schema):
    """Test that an empty list for complex vector raises an error."""
    input_data = {"complexVector": []}
    with pytest.raises(ma.ValidationError) as excinfo:
        complexvector_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


# ---------------------------
# SetOfComplexVectors Field Tests
# ---------------------------
@pytest.fixture
def setofcomplexvectors_schema():
    class Schema(FrontendFormBaseSchema):
        complexVectors = SetOfComplexVectorsField()

    return Schema()


def test_setofcomplexvectors_field_valid(setofcomplexvectors_schema):
    """Test that a valid set of complex vectors is correctly processed."""
    input_data = {
        "complexVectors": [
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
            [[3.0, -2.0], [5.5, 6.0], [-1.5, 9.0]],
            [[11.0, -2.0], [36.5, 0.0], [-1.5, -3.0]],
        ]
    }
    expected_output = {
        "complexVectors": [
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
            [[3.0, -2.0], [5.5, 6.0], [-1.5, 9.0]],
            [[11.0, -2.0], [36.5, 0.0], [-1.5, -3.0]],
        ]
    }
    result = setofcomplexvectors_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_setofcomplexvectors_field_empty_string(setofcomplexvectors_schema):
    """Test that an empty string for set of complex vectors remains unchanged."""
    input_data = {"complexVectors": ""}
    expected_output = {"complexVectors": ""}
    result = setofcomplexvectors_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_setofcomplexvectors_field_missing(setofcomplexvectors_schema):
    """Test that a missing set of complex vectors field returns an empty dictionary."""
    input_data = {}
    expected_output = {}
    result = setofcomplexvectors_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_setofcomplexvectors_field_valid_as_string(setofcomplexvectors_schema):
    """Test that a valid set of complex vectors provided as a JSON string is correctly processed."""
    input_data = {
        "complexVectors": "[[[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],[[3.0, -2.0], [5.5, 6.0], [-1.5, 9.0]],[[11.0, -2.0], [36.5, 0.0], [-1.5, -3.0]]]"
    }
    expected_output = {
        "complexVectors": [
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
            [[3.0, -2.0], [5.5, 6.0], [-1.5, 9.0]],
            [[11.0, -2.0], [36.5, 0.0], [-1.5, -3.0]],
        ]
    }
    result = setofcomplexvectors_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_setofcomplexvectors_field_invalid_not_json(setofcomplexvectors_schema):
    """Test that a non-JSON input raises an error for set of complex vectors."""
    input_data = "invalid_element"
    with pytest.raises(ma.ValidationError) as excinfo:
        setofcomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setofcomplexvectors_field_invalid_empty_list(setofcomplexvectors_schema):
    """Test that an empty list for set of complex vectors raises an error."""
    input_data = {"complexVectors": []}
    with pytest.raises(ma.ValidationError) as excinfo:
        setofcomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setofcomplexvectors_field_invalid_type(setofcomplexvectors_schema):
    """Test that a non-list input for set of complex vectors raises an error."""
    input_data = {"complexVectors": 1}
    with pytest.raises(ma.ValidationError) as excinfo:
        setofcomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setofcomplexvectors_field_invalid_element(setofcomplexvectors_schema):
    """Test that an invalid element within the set of complex vectors raises an error."""
    input_data = {"complexVectors": [[[1.0, -2.0], "invalid_element", [-1.5, 0.0]]]}
    with pytest.raises(ma.ValidationError) as excinfo:
        setofcomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setofcomplexvectors_field_invalid_element_not_a_list(setofcomplexvectors_schema):
    """Test that an not a list element raises an error."""
    input_data = {"complexVectors": "not_a_list"}
    with pytest.raises(ma.ValidationError) as excinfo:
        setofcomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setofcomplexvectors_field_invalid_structure(setofcomplexvectors_schema):
    """Test that an improperly structured set of complex vectors raises an error."""
    input_data = {
        "complexVectors": [[[1.0], [3.5, 4.0]]]
    }  # Missing imaginary part in first element
    with pytest.raises(ma.ValidationError) as excinfo:
        setofcomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


# ---------------------------
# SetOfTwoComplexVectors Field Tests
# ---------------------------
@pytest.fixture
def setoftwocomplexvectors_schema():
    class Schema(FrontendFormBaseSchema):
        complexVectors = SetOfTwoComplexVectorsField()

    return Schema()


def test_setoftwocomplexvectors_field_valid(setoftwocomplexvectors_schema):
    """Test that a valid set of exactly two complex vectors is correctly processed."""
    input_data = {
        "complexVectors": [
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
            [[3.0, -2.0], [5.5, 6.0], [-1.5, 9.0]],
        ]
    }
    expected_output = {
        "complexVectors": [
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
            [[3.0, -2.0], [5.5, 6.0], [-1.5, 9.0]],
        ]
    }
    result = setoftwocomplexvectors_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_setoftwocomplexvectors_field_empty_string(setoftwocomplexvectors_schema):
    """Test that an empty string for set of two complex vectors remains unchanged."""
    input_data = {"complexVectors": ""}
    expected_output = {"complexVectors": ""}
    result = setoftwocomplexvectors_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_setoftwocomplexvectors_field_missing(setoftwocomplexvectors_schema):
    """Test that a missing set of two complex vectors field returns an empty dictionary."""
    input_data = {}
    expected_output = {}
    result = setoftwocomplexvectors_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_setoftwocomplexvectors_field_too_few(setoftwocomplexvectors_schema):
    """Test that providing only one complex vector raises a validation error."""
    input_data = {
        "complexVectors": [
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
        ]
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        setoftwocomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setoftwocomplexvectors_field_too_many(setoftwocomplexvectors_schema):
    """Test that providing more than two complex vectors raises a validation error."""
    input_data = {
        "complexVectors": [
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
            [[1.0, -2.0], [3.5, 4.0], [-1.5, 0.0]],
        ]
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        setoftwocomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setoftwocomplexvectors_field_invalid_empty(setoftwocomplexvectors_schema):
    """Test that an empty list for set of two complex vectors raises an error."""
    input_data = {"complexVectors": []}
    with pytest.raises(ma.ValidationError) as excinfo:
        setoftwocomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setoftwocomplexvectors_field_invalid_element(setoftwocomplexvectors_schema):
    """Test that an invalid element within the set of two complex vectors raises an error."""
    input_data = {"complexVectors": [[[1.0, -2.0], "invalid_element", [-1.5, 0.0]]]}
    with pytest.raises(ma.ValidationError) as excinfo:
        setoftwocomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_setoftwocomplexvectors_field_invalid_structure(setoftwocomplexvectors_schema):
    """Test that an improperly structured set of two complex vectors raises an error."""
    input_data = {"complexVectors": [[[1.0], [3.5, 4.0]]]}  # Missing imaginary part
    with pytest.raises(ma.ValidationError) as excinfo:
        setoftwocomplexvectors_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


# ---------------------------
# QasmMetaDataUrlTuple Field Tests
# ---------------------------
@pytest.fixture
def qasm_metadata_url_tuple_schema():
    class Schema(FrontendFormBaseSchema):
        qasmMetadata = QasmInput()

    return Schema()


def test_qasm_metadata_url_tuple_valid(qasm_metadata_url_tuple_schema):
    """Test that a valid QASM URL tuple is correctly processed."""
    input_data = {
        "qasmMetadata": ["https://example.com/qasm.qasm", {"qubit_intervals": [[4, 5]]}]
    }
    expected_output = {"qasmMetadata": ["https://example.com/qasm.qasm", [[4, 5]]]}
    result = qasm_metadata_url_tuple_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_qasm_metadata_url_tuple_valid_only_qasm(qasm_metadata_url_tuple_schema):
    """Test that a QASM URL without metadata is correctly processed."""
    input_data = {"qasmMetadata": ["https://example.com/qasm.qasm"]}
    expected_output = {"qasmMetadata": ["https://example.com/qasm.qasm"]}
    result = qasm_metadata_url_tuple_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_qasm_metadata_url_tuple_invalid_too_many_elements(
    qasm_metadata_url_tuple_schema,
):
    """Test that a tuple with more than two elements raises an error."""
    input_data = {
        "qasmMetadata": [
            "https://example.com/qasm.qasm",
            {"qubit_intervals": [[4, 5]]},
            "extra",
        ]
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input length" in str(excinfo.value)


def test_qasm_metadata_url_tuple_invalid_format(qasm_metadata_url_tuple_schema):
    """Test that a non-list input raises an error."""
    input_data = {"qasmMetadata": "invalid_string"}
    with pytest.raises(ma.ValidationError) as excinfo:
        qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_qasm_metadata_url_tuple_field_invalid_structure(qasm_metadata_url_tuple_schema):
    """Test that a wrong input raises an error."""
    input_data = {"qasmMetadata": "1"}
    with pytest.raises(ma.ValidationError) as excinfo:
        qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_qasm_metadata_url_tuple_field_invalid_content_not_a_list(
    qasm_metadata_url_tuple_schema,
):
    """Test that a wrong input raises an error."""
    input_data = {
        "qasmMetadata": ["https://example.com/qasm.qasm", {"qubit_intervals": 2}]
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_qasm_metadata_url_tuple_field_invalid_content_not_a_list_of_intervalls(
    qasm_metadata_url_tuple_schema,
):
    """Test that a wrong input raises an error."""
    input_data = {
        "qasmMetadata": ["https://example.com/qasm.qasm", {"qubit_intervals": [2, 2, 2]}]
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_qasm_metadata_url_tuple_field_invalid_content_missing_qubit_intervals(
    qasm_metadata_url_tuple_schema,
):
    """Test that a wrong input raises an error."""
    input_data = {"qasmMetadata": ["https://example.com/qasm.qasm", {}]}
    with pytest.raises(ma.ValidationError) as excinfo:
        qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_qasm_metadata_url_tuple_valid_only_qasm(qasm_metadata_url_tuple_schema):
    """Test that a QASM URL without metadata is correctly processed."""
    input_data = {"qasmMetadata": ["https://example.com/qasm.qasm"]}
    expected_output = {"qasmMetadata": ["https://example.com/qasm.qasm"]}
    result = qasm_metadata_url_tuple_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


# ---------------------------
# SetOfQasmMetaDataUrlTuple Field Tests
# ---------------------------
@pytest.fixture
def set_of_qasm_metadata_url_tuple_schema():
    class Schema(FrontendFormBaseSchema):
        qasmMetadataSet = QasmInputList()

    return Schema()


def test_set_of_qasm_metadata_url_tuple_valid(set_of_qasm_metadata_url_tuple_schema):
    """Test that a valid set of QASM URL tuples is correctly processed."""
    input_data = {
        "qasmMetadataSet": [
            ["https://example.com/qasm1.qasm", {"qubit_intervals": [[4, 5]]}],
            ["https://example.com/qasm2.qasm"],
        ]
    }
    expected_output = {
        "qasmMetadataSet": [
            ["https://example.com/qasm1.qasm", [[4, 5]]],
            ["https://example.com/qasm2.qasm"],
        ]
    }
    result = set_of_qasm_metadata_url_tuple_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_set_of_qasm_metadata_url_tuple_valid_as_string(
    set_of_qasm_metadata_url_tuple_schema,
):
    """Test that a valid set of QASM URL tuples as string is correctly processed without actual URL validation."""
    input_data = {
        "qasmMetadataSet": '[["https://example.com/qasm1.qasm", {"qubit_intervals": [[4, 5]]}],["https://example.com/qasm2.qasm"]]'
    }
    expected_output = {
        "qasmMetadataSet": [
            ["https://example.com/qasm1.qasm", [[4, 5]]],
            ["https://example.com/qasm2.qasm"],
        ]
    }
    result = set_of_qasm_metadata_url_tuple_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_set_of_qasm_metadata_url_tuple_valid_empty_string(
    set_of_qasm_metadata_url_tuple_schema,
):
    """Test that a valid empty string is correctly processed."""
    input_data = {
        "qasmMetadataSet": "",
    }
    expected_output = {"qasmMetadataSet": ""}
    result = set_of_qasm_metadata_url_tuple_schema.load(input_data)
    assert result == expected_output, f"Unexpected result: {result}"


def test_set_of_qasm_metadata_url_tuple__invalid_structure(
    set_of_qasm_metadata_url_tuple_schema,
):
    """Test that a wrong input raises an error."""
    input_data = {
        "qasmMetadataSet": "1",
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        set_of_qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_set_of_qasm_metadata_url_tuple__invalid_structure_not_json(
    set_of_qasm_metadata_url_tuple_schema,
):
    """Test that a wrong input raises an error."""
    input_data = {
        "qasmMetadataSet": "a1 2!",
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        set_of_qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_set_of_qasm_metadata_url_tuple__epty(
    set_of_qasm_metadata_url_tuple_schema,
):
    """Test that a wrong input raises an error."""
    input_data = {
        "qasmMetadataSet": [],
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        set_of_qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid input" in str(excinfo.value)


def test_set_of_qasm_metadata_url_tuple_invalid_endpoint(
    set_of_qasm_metadata_url_tuple_schema,
):
    """Test that a invalid set of QASM URL tuples raises an error with actual URL validation."""
    input_data = {
        "qasmMetadataSet": [
            [["not", "a", "url"], {"qubit_intervals": [[4, 5]]}],
            ["https://example.com/qasm2.qasm"],
        ]
    }
    with pytest.raises(ma.ValidationError) as excinfo:
        set_of_qasm_metadata_url_tuple_schema.load(input_data)
    assert "Invalid QASM/meta tuple" in str(excinfo.value)
