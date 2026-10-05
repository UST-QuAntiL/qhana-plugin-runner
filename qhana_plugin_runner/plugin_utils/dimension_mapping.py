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

"""Module containing helpers to work with ``entity/dimension-mapping`` files."""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .entity_marshalling import ensure_dict, load_entities
from ..requests import get_mimetype, open_url

DIMENSION_MAPPING_DATA_TYPE = "entity/dimension-mapping"
DIMENSION_MAPPING_CONTENT_TYPES = ["application/json"]


def _label(
    mapping_entity: Mapping[str, Any], include_source_dimension: bool = False
) -> Optional[str]:
    source = str(mapping_entity.get("source") or "").strip()
    if not source:
        return None

    label = Path(source).stem or source
    if not include_source_dimension:
        return label

    source_dimension = str(mapping_entity.get("sourceDimension") or "").strip()
    if not source_dimension:
        return label

    return f"{label} ({source_dimension})"


def _labels(
    mapping_entities: Sequence[Any], include_source_dimension: bool = False
) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for entity in mapping_entities:
        dimension = str(entity.get("ID") or "").strip()
        if not dimension:
            continue
        label = _label(entity, include_source_dimension)
        if label:
            labels[dimension] = label

    return labels


def load_dimension_labels(
    url: Optional[str], include_source_dimension: bool = False
) -> Dict[str, str]:
    """Load an ``entity/dimension-mapping`` file and build the dimension labels.

    Args:
        url (Optional[str]): the url of the dimension mapping file
        include_source_dimension (bool): append the column the dimension had in
            the source file to the feature name, e.g. ``"color (dim1)"``

    Returns:
        Dict[str, str]: a mapping from dimension name (``"dim0"``) to its
            label. Empty if no url was given, so that callers can use the
            result unconditionally.
    """
    if not url:
        return {}

    with open_url(url) as response:
        mimetype = get_mimetype(response)
        if mimetype is None:
            raise ValueError("Could not determine mimetype of the dimension mapping.")
        mapping_entities = list(ensure_dict(load_entities(response, mimetype=mimetype)))

    return _labels(mapping_entities, include_source_dimension)
