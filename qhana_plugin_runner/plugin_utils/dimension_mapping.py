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


def _label(
    mapping_entity: Mapping[str, Any], include_source_dimension: bool = False
) -> Optional[str]:
    """Build the display label of a single dimension mapping entity.

    The label is the name of the source feature. The source file name is
    stripped of its extension so that dimensions coming from a zip member
    (``"color.json"``) are labelled like dimensions coming from a plain url
    (``"color"``). Every dimension of a multi dimensional source therefore
    carries the same label, unless ``include_source_dimension`` appends the
    column the dimension had in the source file (``"color (dim1)"``).
    """
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


def dimension_mapping_labels(
    mapping_entities: Sequence[Any], include_source_dimension: bool = False
) -> Dict[str, str]:
    """Build the dimension labels from loaded dimension mapping entities.

    Args:
        mapping_entities (Sequence[Any]): the mapping entities as dicts
        include_source_dimension (bool): append the column the dimension had in
            the source file to the feature name, e.g. ``"color (dim1)"``. Use
            this where the labels have to tell the dimensions of one multi
            dimensional feature apart.

    Returns:
        Dict[str, str]: a mapping from dimension name (``"dim0"``) to its label
    """
    labels: Dict[str, str] = {}
    for entity in mapping_entities:
        dimension = str(entity.get("ID") or "").strip()
        if not dimension:
            continue
        label = _label(entity, include_source_dimension)
        if label:
            labels[dimension] = label

    return labels


def load_dimension_mapping(
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

    return dimension_mapping_labels(mapping_entities, include_source_dimension)


def dimension_labels(names: Sequence[str], labels: Mapping[str, str]) -> List[str]:
    """Label the given dimension names, falling back to the name itself.

    Args:
        names (Sequence[str]): the dimension names to label
        labels (Mapping[str, str]): labels from ``load_dimension_mapping``

    Returns:
        List[str]: the label for every given dimension name
    """
    return [labels.get(name, name) for name in names]
