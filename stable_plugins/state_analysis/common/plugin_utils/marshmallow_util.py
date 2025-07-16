import json

import marshmallow as ma
from qhana_plugin_runner.api.util import FileUrl

"""
This module defines custom Marshmallow fields for validating and deserializing
complex numerical data structures, including complex numbers, vectors, and sets of vectors.

Some fields are designed to return an empty string ("") when receiving one.
This is necessary because the frontend sends empty strings for optional fields when no input is provided.
Handling this ensures proper backend processing and prevents unintended validation errors.
"""


class ToleranceField(ma.fields.Float):
    """
    A custom Float field for handling tolerance values.

    If the input is an empty string (""), it returns an empty string.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value in (None, ""):
            return ""
        number = super()._validated(value)
        if number < 0:
            raise ma.ValidationError(
                f"Invalid input. Tolerance cannot be smaller than 0 but was {number}."
            )
        return number


class ComplexNumberField(ma.fields.Field):
    """
    Field for deserializing a complex number represented as [real, imag].
    """

    def _deserialize(self, value, attr, data, **kwargs):
        # If value is a string, attempt to parse it as JSON.
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ma.ValidationError(
                    f"Invalid input. Expected a JSON string representing a list, but could not parse: {value}"
                )

        # Ensure the value is a list.
        if not isinstance(value, (list, tuple)):
            raise ma.ValidationError(
                f"Invalid input. Expected a list with two elements [real, imag], but got {value}"
            )
        if len(value) != 2:
            raise ma.ValidationError(
                f"Invalid input length. Expected two elements [real, imag], but got {value}"
            )
        try:
            real, imag = float(value[0]), float(value[1])
            return [real, imag]
        except (ValueError, TypeError):
            raise ma.ValidationError(
                f"Invalid numbers provided. Real and imaginary parts must be valid floats, but got {value}"
            )


class ComplexVectorField(ma.fields.Field):
    """
    Field for deserializing a vector of complex numbers.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value in (None, ""):
            return ""

        # If value is a string, attempt to parse it as JSON.
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ma.ValidationError(
                    f"Invalid input. Expected a JSON string representing a list, but could not parse: {value}"
                )

        if not isinstance(value, list):
            raise ma.ValidationError(
                f"Invalid input. Expected a list of complex number representations, but got: {value}"
            )
        if not value:
            raise ma.ValidationError(
                "Invalid input. The list of complex number representations cannot be empty."
            )

        output = []
        for comp_num in value:
            try:
                deserialized_num = ComplexNumberField().deserialize(comp_num)
                output.append(deserialized_num)
            except ma.ValidationError as e:
                raise ma.ValidationError(
                    f"Invalid complex number in vector: {comp_num}. Error: {e.messages}"
                )
        return output


class SetOfComplexVectorsField(ma.fields.Field):
    """
    Field for deserializing a set of complex vectors.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value in (None, ""):
            return ""

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ma.ValidationError(
                    f"Invalid input. Expected a JSON string representing a list, but could not parse: {value}"
                )

        if not isinstance(value, list):
            raise ma.ValidationError(
                f"Invalid input. Expected a list of complex vector representations, but got: {value}"
            )
        if not value:
            raise ma.ValidationError(
                "Invalid input. The list of complex vectors cannot be empty."
            )

        output = []
        for comp_vec in value:
            try:
                deserialized_vec = ComplexVectorField().deserialize(comp_vec)
                output.append(deserialized_vec)
            except ma.ValidationError as e:
                raise ma.ValidationError(
                    f"Invalid complex vector in set: {comp_vec}. Error: {e.messages}"
                )
        return output


class SetOfTwoComplexVectorsField(ma.fields.Field):
    """
    Field for deserializing a set of exactly two complex vectors.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value in (None, ""):
            return ""

        value = SetOfComplexVectorsField().deserialize(value)

        if len(value) != 2:
            raise ma.ValidationError(
                f"Invalid input. Expected a list of exactly 2 vectors, but got {value}."
            )
        return value


class QasmInput(ma.fields.Field):
    """
    Deserializes a list containing one or two strings:
      - The first element is a URL to a QASM file (required).
      - The second element is a json stirng with meta data (optional, may be None).
    """

    def _deserialize(self, value, attr, data, **kwargs):

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ma.ValidationError(
                    f"Invalid input. Expected a JSON string representing a list, but could not parse: {value}"
                )

        if not isinstance(value, (list, tuple)):
            raise ma.ValidationError(f"Invalid input. Expected a list, but got: {value}")

        if not (1 <= len(value) <= 2):
            raise ma.ValidationError(
                f"Invalid input length. Expected 1 or 2 elements, but got: {value}"
            )

        qasm_url = FileUrl()._deserialize(value[0], attr, data, **kwargs)

        if len(value) == 2:
            metadata_json = value[1]
            if metadata_json is not None:

                if isinstance(metadata_json, dict) and "qubit_intervals" in metadata_json:
                    qubit_intervals = metadata_json["qubit_intervals"]

                    if isinstance(qubit_intervals, list):

                        if all(
                            isinstance(interval, list)
                            and len(interval) == 2
                            and all(isinstance(num, int) for num in interval)
                            for interval in qubit_intervals
                        ):
                            return [qasm_url, qubit_intervals]
                        else:
                            raise ma.ValidationError(
                                "Invalid input: 'qubit_intervals' must be a list of lists, each containing exactly two integers."
                            )
                    else:
                        raise ma.ValidationError(
                            "Invalid input: 'qubit_intervals' must be a list."
                        )

            raise ma.ValidationError(
                "Invalid input: json must contain 'qubit_intervals'."
            )
        else:
            return [qasm_url]


class QasmInputList(ma.fields.Field):
    """
    Deserializes a list of QasmMetaDataUrlTuple entries.
    Each entry must be a list with one or two URL strings.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value in (None, ""):
            return ""

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ma.ValidationError(
                    f"Invalid input. Expected a JSON string representing a list, but could not parse: {value}"
                )
        if not isinstance(value, list):
            raise ma.ValidationError(
                f"Invalid input. Expected a list of QASM/meta 'tuple (as a list)' entries, but got: {value}"
            )

        if not value:
            raise ma.ValidationError(
                "Invalid input. The list of QASM and metadata URL entries cannot be empty."
            )

        output = []
        for tup in value:
            try:
                deserialized_tup = QasmInput().deserialize(tup)
                output.append(deserialized_tup)
            except ma.ValidationError as e:
                raise ma.ValidationError(
                    f"Invalid QASM/meta tuple (as a list)': {tup}. Error: {e.messages}"
                )
        return output
