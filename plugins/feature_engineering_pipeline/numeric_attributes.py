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
from io import StringIO
from tempfile import SpooledTemporaryFile
from typing import Any, cast
from zipfile import ZipFile

from qhana_plugin_runner.plugin_utils.entity_marshalling import save_entities


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


def collect_values(entities: list[dict[str, Any]], attribute: str) -> list[float | None]:
    """Return the parsed values of ``attribute`` in entity order.

    A missing value is ``None``.
    """
    return [
        _parse_float(entity["ID"], attribute, entity.get(attribute))
        for entity in entities
    ]


def require_complete_column(
    entity_ids: list[str], values: list[float | None], attribute: str
) -> list[float]:
    """Return the column values, one per entity.

    Raises a ``ValueError`` if ``attribute`` has no value for an entity. The
    message names at most five of the affected entities.
    """
    missing = [entity_id for entity_id, value in zip(entity_ids, values) if value is None]
    if missing:
        names = ", ".join(repr(entity_id) for entity_id in missing[:5])
        if len(missing) > 5:
            names += ", ..."
        raise ValueError(
            f"The attribute '{attribute}' has no numeric value for the entities {names}."
        )
    return cast(list[float], values)


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
