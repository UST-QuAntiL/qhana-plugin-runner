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

"""Module containing helpers to work with ``entity/dimension-mapping`` files.

Plugins like ``vector-concat`` renumber the columns of the vector files
they merge, which loses the connection between an output dimension and the
feature it came from. A dimension mapping file restores that connection: it
contains one entity per output dimension, with the name of the dimension in
``ID`` and the input file and column it originated from in ``source`` and
``sourceDimension``.

See :ref:`data-formats/examples/entities:entity/dimension-mapping`.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..requests import get_mimetype, open_url
from .entity_marshalling import ensure_dict, load_entities

DIMENSION_MAPPING_DATA_TYPE = "entity/dimension-mapping"
"""The data type of dimension mapping files."""

DIMENSION_MAPPING_CONTENT_TYPES = ["application/json"]
"""The content types a dimension mapping file may be serialized in."""


def entity_dimension_names(entities: Sequence[Any]) -> List[str]:
    """Get the dimension names of a loaded ``entity/vector`` file.

    The names are returned in the same order in which
    :py:func:`~qhana_plugin_runner.plugin_utils.entity_marshalling.ensure_array`
    yields the corresponding values, i.e. sorted by attribute name for entities
    loaded as dicts and in field order for entities loaded as named tuples.

    Args:
        entities (Sequence[Any]): the entities as returned by ``load_entities``

    Returns:
        List[str]: the dimension names, empty if there are no entities
    """
    if not entities:
        return []

    first = entities[0]

    if isinstance(first, dict):
        return sorted(key for key in first if key not in ("ID", "href"))

    fields = list(first._fields)
    return fields[2:] if "href" in fields else fields[1:]


def _source_key(mapping_entity: Mapping[str, Any]) -> tuple:
    return (
        mapping_entity.get("source"),
        mapping_entity.get("sourceUrl"),
        mapping_entity.get("zipMember"),
    )


def _label(mapping_entity: Mapping[str, Any], is_only_dimension: bool) -> Optional[str]:
    """Build the display label of a single dimension mapping entity.

    The source file name is stripped of its extension so that dimensions
    coming from a zip member (``"color.json"``) are labelled like dimensions
    coming from a plain url (``"color"``). The source dimension is appended to
    keep the labels unique when a source contributes more than one dimension.
    """
    source = str(mapping_entity.get("source") or "").strip()
    if not source:
        return None

    name = Path(source).stem or source

    if is_only_dimension:
        return name

    source_dimension = str(mapping_entity.get("sourceDimension") or "").strip()
    if not source_dimension:
        return name

    return f"{name} ({source_dimension})"


def dimension_mapping_labels(mapping_entities: Sequence[Any]) -> Dict[str, str]:
    """Build the dimension labels from loaded dimension mapping entities.

    Args:
        mapping_entities (Sequence[Any]): the mapping entities as dicts

    Returns:
        Dict[str, str]: a mapping from dimension name (``"dim0"``) to its label
    """
    dimensions_per_source = Counter(_source_key(entity) for entity in mapping_entities)

    labels: Dict[str, str] = {}
    for entity in mapping_entities:
        dimension = str(entity.get("ID") or "").strip()
        if not dimension:
            continue
        label = _label(entity, dimensions_per_source[_source_key(entity)] == 1)
        if label:
            labels[dimension] = label

    return labels


def load_dimension_mapping(url: Optional[str]) -> Dict[str, str]:
    """Load an ``entity/dimension-mapping`` file and build the dimension labels.

    Args:
        url (Optional[str]): the url of the dimension mapping file

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

    return dimension_mapping_labels(mapping_entities)


def dimension_labels(names: Sequence[str], labels: Mapping[str, str]) -> List[str]:
    """Label the given dimension names, falling back to the name itself.

    Args:
        names (Sequence[str]): the dimension names to label
        labels (Mapping[str, str]): labels from ``load_dimension_mapping``

    Returns:
        List[str]: the label for every given dimension name
    """
    return [labels.get(name, name) for name in names]
