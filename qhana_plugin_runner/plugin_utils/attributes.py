# Copyright 2021 QHAna plugin runner contributors.
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

"""Module containing helpers to work with entity attribute data and attribute metadata entities."""

from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

_ATTR_MAPPING = (
    ("ID", "ID"),
    ("attribute_type", "type"),
    ("title", "title"),
    ("description", "description"),
    ("multiple", "multiple"),
    ("ordered", "ordered"),
    ("separator", "separator"),
    ("ref_target", "refTarget"),
    ("schema", "schema"),
)

_ATTR_MAPPING_SER = dict(_ATTR_MAPPING)
_ATTR_MAPPING_DE = dict([pair[::-1] for pair in _ATTR_MAPPING])

CONSIDERED_TRUE: Set[str] = {"1", "true", "t", "yes", "y", "on"}
CONSIDERED_FALSE: Set[str] = {"0", "false", "f", "no", "n", "off", "null", "nil", "none"}


@dataclass
class AttributeMetadata:
    """Dataclass for entity attribute metadata."""

    ID: str
    attribute_type: str
    title: str
    description: str = ""
    multiple: bool = False
    ordered: bool = False
    separator: str = ";"
    ref_target: Optional[str] = None
    schema: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(metadata: Dict[str, Union[str, bool]]) -> "AttributeMetadata":
        """Create a AttributeMetadata instance from an entity dict."""
        mapped = {}
        extra = {}
        for key, value in metadata.items():
            if key in _ATTR_MAPPING_DE:
                mapped[_ATTR_MAPPING_DE[key]] = value
            else:
                extra[key] = value
        if "title" not in mapped:
            mapped["title"] = mapped["ID"]
        for bool_attr in ("multiple", "ordered"):
            if bool_attr in mapped:
                mapped[bool_attr] = parse_bool(mapped[bool_attr])
        return AttributeMetadata(**mapped, extra=extra)

    def to_dict(self):
        """Get the attribute metadata as entity dict."""
        mapped = {}
        # early update as actual data should overwrite erroneous extra attributes
        mapped.update(self.extra)
        for attr, mapped_attr in _ATTR_MAPPING_SER.items():
            mapped[mapped_attr] = getattr(self, attr)
        return mapped


def parse_bool(value: Union[str, bool]) -> bool:
    """Parse a string value into a boolean.

    Uses the sets ``CONSIDERED_TRUE`` and ``CONSIDERED_FALSE`` to determine the boolean value of the string.

    Args:
        value (Union[str, bool]): the string to parse (is converted to lowercase and stripped of surrounding whitespace)

    Raises:
        ValueError: if the string cannot reliably be determined true or false

    Returns:
        bool: the parsed result
    """
    if value is True or value is False:
        return value
    val = value.strip().lower()
    if val in CONSIDERED_TRUE:
        return True
    if val in CONSIDERED_FALSE:
        return False
    raise ValueError(f"Value {value} is not compatible with boolean!")


def parse_optional_value(value: str, deserialize: Callable[[str], Any]) -> Any:
    """Serialize optional values with the given serializer."""
    if value == "":
        return None
    return deserialize(value)


T = TypeVar("T")


def parse_multi(
    value: str,
    deserialize: Callable[[str], Any],
    separator: str = ";",
    collection: Type[T] = list,
) -> T:
    """Parse a multi value string into a collection of parsed values.

    Args:
        value (str): the string to parse
        deserialize (Callable[[str], Any]): the deserialize function to parse single values
        separator (str, optional): the separator sequence that separates single values. Must not be part of the values! Defaults to ";".
        collection (Type[T], optional): the collection class to use. Defaults to list.

    Returns:
        T: the collection of parsed values
    """
    if value == "":
        return collection()
    return collection(deserialize(val) for val in value.split(separator))


def default_serialize(value: Any) -> str:
    """Default value serializer.

    ``None`` -> ``""``
    ``value: Union[bool, int, float, str]`` -> ``str(value)``
    ``value: Any`` -> ``repr(value)``
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return str(value)
    if value is None:
        return ""
    return repr(value)


def serialize_multi(
    value: Optional[Iterable[Any]], serialize: Callable[[Any], str], separator: str = ";"
) -> str:
    """Serialize a list or set of values into a string.

    Each value is serialized with the ``serializer`` and the values are joined with the ``separator``.

    Args:
        value (Iterable[Any]|None): the list or set of values to serialize
        serialize (Callable[[Any], str]): the serializer to use for single values
        separator (str, optional): the seperator to use between values. Must not be part of the serialized values! Defaults to ";".

    Returns:
        str: the string of serialized items.
    """
    if value is None:
        return ""
    return separator.join(serialize(val) for val in value)


# Map mapping from attribute metadata attribute type to serializer function
SERIALIZER_MAP: Dict[str, Callable[[Any], str]] = {
    "default": default_serialize,
    "string": default_serialize,
    "str": default_serialize,
    "integer": default_serialize,
    "int": default_serialize,
    "float": default_serialize,
    "double": default_serialize,
    "number": default_serialize,
    "bool": default_serialize,
    "boolean": default_serialize,
}


# Map mapping from attribute metadata attribute type to de-serializer function
DESERIALIZER_MAP: Dict[str, Callable[[str], Any]] = {
    "default": lambda x: x,
    "string": lambda x: x,
    "str": lambda x: x,
    "integer": partial(parse_optional_value, deserialize=int),
    "int": partial(parse_optional_value, deserialize=int),
    "float": partial(parse_optional_value, deserialize=float),
    "double": partial(parse_optional_value, deserialize=float),
    "number": partial(parse_optional_value, deserialize=float),
    "bool": parse_bool,
    "boolean": parse_bool,
}


def _get_serializer(meta: Optional[AttributeMetadata]) -> Callable[[Any], str]:
    """Get a serializer function from the given attribute metadata."""
    serializer = None if meta is None else SERIALIZER_MAP.get(meta.attribute_type)
    if serializer is None:
        serializer = SERIALIZER_MAP.get("default", repr)
    if meta and meta.multiple:
        return partial(serialize_multi, serialize=serializer, separator=meta.separator)
    return serializer


def _get_deserializer(meta: Optional[AttributeMetadata]) -> Callable[[str], Any]:
    """Get a de-serializer function from the given attribute metadata."""
    deserializer = None if meta is None else DESERIALIZER_MAP.get(meta.attribute_type)
    if deserializer is None:
        deserializer = DESERIALIZER_MAP.get("default", lambda x: x)
    if meta and meta.multiple:
        collection = list if meta.ordered else set
        return partial(
            parse_multi,
            deserialize=deserializer,
            separator=meta.separator,
            collection=collection,
        )
    return deserializer


def parse_attribute_metadata(
    metadata_entities: Iterable[Union[Dict[str, Any]]]
) -> Dict[str, AttributeMetadata]:
    """Parse a list or stream of entities into an attribute metadata dict mapping from attribute name to ``AttributeMetadata`` instance."""
    attribute_metadata = {}
    for metadata in metadata_entities:
        attr = AttributeMetadata.from_dict(metadata=metadata)
        attribute_metadata[attr.ID] = attr
    return attribute_metadata


def tuple_serializer(
    attributes: Sequence[str],
    attribute_metadata: Dict[str, AttributeMetadata],
    tuple_: Callable[[Iterable], tuple] = tuple,
) -> Callable[[Tuple[Any, ...]], Tuple[str, ...]]:
    """Create a tuple serializer that serializes the values of a tuple into strings based on the attribute metadata.

    The returned serializer serializes the attributes of a single entity tuple to a string.
    It can be used in a list or generator comprehension or with ``map``.

    Args:
        attributes (Sequence[str]): the attribute names of the components of the tuple (must be the same for all tuples!)
        attribute_metadata (Dict[str, AttributeMetadata]): the attribute metadata dict (see :py:func:`~qhana_plugin_runner.plugin_utils.attributes.parse_attribute_metadata`)
        tuple_ (Callable[[Iterable], tuple]): the tuple type to use (should be a ``namedtuple._make`` function). Defaults to :py:func:`tuple`

    Returns:
        Callable[[Tuple[Any, ...]], Tuple[str, ...]]: the serializer function
    """
    serializer: List[Callable[[Any], str]] = []
    for attr in attributes:
        meta = attribute_metadata.get(attr)
        serializer.append(_get_serializer(meta))

    def _tuple_serializer(entity: Tuple[Any, ...]) -> Tuple[str, ...]:
        return tuple_(ser(attr) for ser, attr in zip(serializer, entity))

    return _tuple_serializer


def tuple_deserializer(
    attributes: Sequence[str],
    attribute_metadata: Dict[str, AttributeMetadata],
    tuple_: Callable[[Iterable], tuple] = tuple,
) -> Callable[[Tuple[str, ...]], Tuple[Any, ...]]:
    """Create a tuple de-serializer that parses the values of a tuple from strings based on the attribute metadata.

    The returned de-serializer parses the attributes of a single entity tuple from their string values.
    It can be used in a list or generator comprehension or with ``map``.

    Args:
        attributes (Sequence[str]): the attribute names of the components of the tuple (must be the same for all tuples!)
        attribute_metadata (Dict[str, AttributeMetadata]): the attribute metadata dict (see :py:func:`~qhana_plugin_runner.plugin_utils.attributes.parse_attribute_metadata`)
        tuple_ (Type[tuple]): the tuple type to use (should be a ``namedtuple._make`` function). Defaults to :py:func:`tuple`

    Returns:
        Callable[[Tuple[str, ...]], Tuple[Any, ...]]: the de-serializer function
    """
    deserializer: List[Callable[[str], Any]] = []
    for attr in attributes:
        meta = attribute_metadata.get(attr)
        deserializer.append(_get_deserializer(meta))

    def _tuple_deserializer(entity: Tuple[str, ...]) -> Tuple[Any, ...]:
        return tuple_(ser(attr) for ser, attr in zip(deserializer, entity))

    return _tuple_deserializer


def dict_serializer(
    attributes: Iterable[str],
    attribute_metadata: Dict[str, AttributeMetadata],
    in_place: bool = True,
) -> Callable[[Dict[str, Any]], Dict[str, str]]:
    """Create a dict serializer that serializes the values of a dict into strings based on the attribute metadata.

    The returned serializer serializes the attributes of a single entity dict to a string.
    It can be used in a list or generator comprehension or with ``map``.

    Args:
        attributes (Iterable[str]): the attribute names of the components of the dict (must be the same for all dicts!)
        attribute_metadata (Dict[str, AttributeMetadata]): the attribute metadata dict (see :py:func:`~qhana_plugin_runner.plugin_utils.attributes.parse_attribute_metadata`)
        in_place (bool, optional): if True the serialized values will replace the old values in the dict, if False a new dict is used. Defaults to True.

    Returns:
        Callable[[Dict[str, Any]], Dict[str, str]]: the serializer function
    """
    serializer: Dict[str, Callable[[Any], str]] = {}
    for attr in attributes:
        meta = attribute_metadata.get(attr)
        serializer[attr] = _get_serializer(meta)

    if in_place:

        def _dict_serializer_in_place(entity: Dict[str, Any]) -> Dict[str, str]:
            for attr in entity:
                entity[attr] = serializer[attr](entity[attr])
            return entity

        return _dict_serializer_in_place

    def _dict_serializer(entity: Dict[str, Any]) -> Dict[str, str]:
        return {attr: serializer[attr](value) for attr, value in entity.items()}

    return _dict_serializer


def dict_deserializer(
    attributes: Iterable[str],
    attribute_metadata: Dict[str, AttributeMetadata],
    in_place: bool = True,
) -> Callable[[Dict[str, str]], Dict[str, Any]]:
    """Create a dict de-serializer that parses the values of a dict from strings based on the attribute metadata.

    The returned de-serializer parses the attributes of a single entity dict from their string values.
    It can be used in a list or generator comprehension or with ``map``.

    Args:
        attributes (Iterable[str]): the attribute names of the components of the dict (must be the same for all dicts!)
        attribute_metadata (Dict[str, AttributeMetadata]): the attribute metadata dict (see :py:func:`~qhana_plugin_runner.plugin_utils.attributes.parse_attribute_metadata`)
        in_place (bool, optional): if True the de-serialized values will replace the old values in the dict, if False a new dict is used. Defaults to True.

    Returns:
        Callable[[Dict[str, str]], Dict[str, Any]]: the de-serializer function
    """
    deserializer: Dict[str, Callable[[str], Any]] = {}
    default = DESERIALIZER_MAP["default"]
    for attr in attributes:
        meta = attribute_metadata.get(attr)
        deserializer[attr] = _get_deserializer(meta)

    if in_place:

        def _dict_deserializer_in_place(entity: Dict[str, str]) -> Dict[str, Any]:
            for attr in entity:
                entity[attr] = deserializer[attr](entity[attr])
            return entity

        return _dict_deserializer_in_place

    def _dict_deserializer(entity: Dict[str, str]) -> Dict[str, Any]:
        return {attr: deserializer[attr](value) for attr, value in entity.items()}

    return _dict_deserializer
