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

import math
from collections.abc import Sequence
from io import StringIO
from tempfile import SpooledTemporaryFile
from typing import Any
from zipfile import ZipFile

from qhana_plugin_runner.plugin_utils.attributes import AttributeMetadata
from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities

Vector = list[float | None]


def _is_missing(raw: Any) -> bool:
    if raw is None:
        return True
    if isinstance(raw, float) and math.isnan(raw):
        return True
    if isinstance(raw, str):
        stripped = raw.strip()
        return not stripped or stripped.lower() == "nan"
    return False


def _parse_float(entity_id: str, attribute: str, raw: Any) -> float | None:
    if _is_missing(raw):
        return None
    if isinstance(raw, (bool, list, tuple, set, dict)):
        raise ValueError(
            f"Entity '{entity_id}' has the non-numeric value {raw!r} "
            f"for the attribute '{attribute}'."
        )
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"Entity '{entity_id}' has the non-numeric value {raw!r} "
            f"for the attribute '{attribute}'."
        ) from None


def parse_single_value(entity_id: str, attribute: str, raw: Any) -> float | None:
    """Parse the raw value of a single-valued numeric attribute.

    ``None``, an empty or whitespace string and ``nan`` give ``None``. A list
    or a value that cannot be parsed raises a ``ValueError``.
    """
    return _parse_float(entity_id, attribute, raw)


def parse_vector(
    entity_id: str, attribute: str, raw: Any, separator: str
) -> Vector | None:
    """Parse the raw value of a multi-valued numeric attribute as a vector.

    A string is split on ``separator``. An empty part or ``nan`` becomes
    ``None`` at that position. Returns ``None`` if there is no value at all,
    which includes a vector that consists of missing positions only.
    """
    if _is_missing(raw):
        return None
    if isinstance(raw, str):
        parts: Sequence[Any] = raw.split(separator)
    elif isinstance(raw, (list, tuple)):
        parts = raw
    else:
        parts = [raw]
    vector = [_parse_float(entity_id, attribute, part) for part in parts]
    if all(value is None for value in vector):
        return None
    return vector


def collect_values(
    entities: list[dict[str, Any]], attribute: str, metadata: AttributeMetadata
) -> list:
    """Return the parsed values of ``attribute`` in entity order.

    Multi-valued attributes give vectors, single-valued attributes give floats.
    Missing values are ``None`` in both cases. All vectors must have the same
    length.
    """
    if not metadata.multiple:
        return [
            parse_single_value(entity["ID"], attribute, entity.get(attribute))
            for entity in entities
        ]

    vectors = [
        parse_vector(entity["ID"], attribute, entity.get(attribute), metadata.separator)
        for entity in entities
    ]
    lengths = {len(vector) for vector in vectors if vector is not None}
    if len(lengths) > 1:
        raise ValueError(
            f"The vectors of the attribute '{attribute}' have different lengths "
            f"{sorted(lengths)}. All entities must have the same number of values."
        )
    return vectors


def normalized_column(values: list[float | None]) -> list[float]:
    """Fill missing values with the mean, then scale the values to [0, 1].

    A constant attribute becomes 0 for every entity. Raises a ``ValueError``
    if no value is known.
    """
    known = [value for value in values if value is not None]
    if not known:
        raise ValueError("The attribute has no numeric value for any entity.")
    mean = sum(known) / len(known)
    filled = [mean if value is None else value for value in values]
    low, high = min(filled), max(filled)
    if high == low:
        return [0.0 for _ in filled]
    return [(value - low) / (high - low) for value in filled]


def entities_zip(members: dict[str, list[dict]]) -> SpooledTemporaryFile:
    """Write one ``{attribute}.json`` entity file per member into a zip."""
    zip_buffer = SpooledTemporaryFile(mode="wb")
    with ZipFile(zip_buffer, "w") as zip_file:
        for attribute, entities in members.items():
            with StringIO() as file:
                save_entities(entities, file, "application/json")
                zip_file.writestr(f"{attribute}.json", file.getvalue())
    zip_buffer.seek(0)
    return zip_buffer
