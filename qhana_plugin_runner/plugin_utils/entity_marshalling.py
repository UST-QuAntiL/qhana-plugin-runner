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

"""Module containing helpers to marshall and unmarshall entities into csv or json files."""

from collections import namedtuple
from csv import QUOTE_ALL, Dialect, reader, register_dialect, writer
from json import dump, dumps, loads
from json.decoder import JSONDecoder
from keyword import iskeyword
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Text,
    TextIO,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from unicodedata import category, normalize

from typing_extensions import Protocol


class EntityTupleMixin:
    """A mixin class to provide entity metadata (e.g. attribute names) and
    some helper functions to a namedtuple class.

    Use the helper method ``get_entity_tuple_class`` to create a new entity tuple class.
    """

    entity_attributes: ClassVar[Sequence[str]]
    """The list of attribute names."""
    _attribute_to_index: ClassVar[Dict[str, int]]
    """Helper map to convert an attribute name to its index fast."""

    def __init_subclass__(cls) -> None:
        assert issubclass(cls, tuple), f"Class {cls} must also inherit from a namedtuple!"
        assert cls.entity_attributes, "Entity Attributes must be present!"
        cls._attribute_to_index = {a: i for i, a in enumerate(cls.entity_attributes)}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def get(self, key: Union[str, int], default=None):
        if isinstance(key, str):
            try:
                key = self._attribute_to_index[key]
            except KeyError:
                return default
        return self[key]

    def as_dict(self) -> Dict[str, Any]:
        """Convert the entity tuple to a dict."""
        assert isinstance(self, tuple)
        return dict(zip(self.entity_attributes, self))

    @classmethod
    def from_dict(cls, **kwargs):
        """Create an entity tuple from key=value mapping (using keyword arguments)."""
        assert issubclass(cls, tuple)
        values: Sequence[Any] = tuple(map(kwargs.get, cls.entity_attributes))
        return cls(*values)

    @classmethod
    def from_iter(cls, iterable: Sequence):
        """Create an entity tuple from any iterable."""
        return cls(*iterable)


_ENTITY_TYPE_TUPLE_CLASSES: Dict[Tuple[str, ...], Type[NamedTuple]] = {}  # TODO cache invalidation (use lru cache?)


def get_entity_tuple_class(
    attributes: Sequence[str], name: str = "Entity"
) -> Type[NamedTuple]:
    """Get an entity tuple class.

    Caches the classes based on attributes and provided class name.

    Args:
        attributes (Sequence[str]): the list of entity attributes
        name (str, optional): the name to use for creating the new type. Defaults to "Entity".

    Returns:
        Type[NamedTuple]: the created entity tuple class
    """
    attributes = tuple(attributes)  # make an immutable copy

    key = (name, *attributes)

    if key in _ENTITY_TYPE_TUPLE_CLASSES:
        return _ENTITY_TYPE_TUPLE_CLASSES[key]

    EntityType: Type[NamedTuple] = type(
        name,
        (EntityTupleMixin, namedtuple(name, attributes, rename=True)),
        {"entity_attributes": attributes},
    )

    _ENTITY_TYPE_TUPLE_CLASSES[key] = EntityType
    return EntityType


class ResponseLike(Protocol):
    """A protocol of the minimal interface of :py:class:`~requests.Response` that can be used to read responses."""

    def json(
        self,
        *,
        cls: Optional[Type[JSONDecoder]] = None,
        object_hook: Optional[Callable[[Dict[Any, Any]], Any]] = None,
        parse_float: Optional[Callable[[str], Any]] = None,
        parse_int: Optional[Callable[[str], Any]] = None,
        parse_constant: Optional[Callable[[str], Any]] = None,
        object_pairs_hook: Optional[Callable[[List[Tuple[Any, Any]]], Any]] = None,
        **kwds: Any,
    ) -> Any:
        ...

    def iter_lines(
        self,
        chunk_size: int = 512,
        decode_unicode: bool = False,
        delimiter: Optional[Union[Text, bytes]] = None,
    ) -> Iterator[Any]:
        ...


def entity_attribute_sort_key(attribute_name: str):
    """A sort key function that can be used to sort keys from a dictionary before passing
    them to :py:func:`save_entities` or creating a NamedTuple entity class."""
    if attribute_name in ("ID", "GRAPH_ID"):
        return (0, attribute_name)
    elif attribute_name in ("href", "source", "target", "entities", "relations"):
        return (1, attribute_name)
    return (2, attribute_name)


class DefaultDialect(Dialect):
    """The default csv dialect used to serialize entities.

    Quotes all fields to correctly escape special unicode characters.

    Registered under the name ``"default"``.
    """

    delimiter = ","
    doublequote = True
    quotechar = '"'
    escapechar = None
    skipinitialspace = False
    lineterminator = "\r\n"
    quoting = QUOTE_ALL  # quote all to ensure all unicode is correctly escaped
    strict = False


register_dialect("default", DefaultDialect)


def ensure_dict(
    items: Iterable[Union[Dict[str, Any], NamedTuple]]
) -> Generator[Dict[str, Any], None, None]:
    """Ensure that all entities in an iterable are dicts.

    Args:
        items (Iterable[Union[Dict[str, Any], NamedTuple]]): the input iterable

    Yields:
        Generator[Dict[str, Any], None, None]: the output iterable
    """
    for item in items:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, EntityTupleMixin):
            yield item.as_dict()
        else:
            yield item._asdict()


T = TypeVar("T", tuple, Tuple, NamedTuple)


def ensure_tuple(
    items: Iterable[Union[Dict[str, Any], NamedTuple]], tuple_: Callable[..., T]
) -> Generator[Union[NamedTuple, T], None, None]:
    """Ensure that all entities in an iterable are namedtuples.

    Args:
        items (Iterable[Union[Dict[str, Any], NamedTuple]]): the input iterable
        tuple_ (Callable[..., T]): The namedtuple class to construct tuples from dicts with

    Yields:
        Generator[NamedTuple, None, None]: the output iterable
    """
    if issubclass(tuple_, EntityTupleMixin):
        tuple_ = tuple_.from_dict
    for item in items:
        if isinstance(item, dict):
            yield tuple_(**item)
        else:
            yield item


def load_entities(
    file_: ResponseLike,
    mimetype: str,
    csv_dialect: str = "default",
    tuple_: Optional[Callable[[Iterable[Any]], T]] = None,
    process_csv_header: Optional[Callable[[Sequence[str]], Sequence[str]]] = None,
) -> Generator[Union[Dict[str, Any], T], None, None]:
    """Load entities from a :py:class:`~requests.Response` like object.

    Attributes of entities are either deserialized as json or as strings (csv).

    If the mimetype is "text/csv" this method returns a stream of namedtuples.
    For json dicts are returned. Use the generator functions :py:func:`~qhana_plugin_runner.plugin_utils.entity_marshalling.ensure_dict`
    and :py:func:`~qhana_plugin_runner.plugin_utils.entity_marshalling.ensure_tuple`
    to always convert items in the result stream to dicts or tuples.

    For csv files this method produces namedtuples as output.
    For this to work all column names must be valid python identifiers and will be normalized with :py:func:`normalize_attribute_name`.
    This behaviour can be overwritten with the ``process_csv_header`` callback.
    If the callback is set then the header names will not be normalized with :py:func:`normalize_attribute_name`!

    Args:
        file_ (ResponseLike): the object to load the entities from
        mimetype (str): the mime type to use for deserialization (supported mimetypes: "application/json", "application/X-lines+json" and "text/csv")
        csv_dialect (str, optional): the csv dialect to use (only used with csv mimetype). Defaults to "default".
        tuple_ (Optional[Type[NamedTuple]], optional): the namedtuple class to use (only used with csv mimetype). Defaults to None.
        process_csv_header (Optional[Callable[[Sequence[str]], Sequence[str]]]): a callback used to process the csv header. Defaults to None.

    Raises:
        ValueError: For unknown mimetypes

    Yields:
        Generator[Union[Dict[str, Any], NamedTuple], None, None]: a stream of deserialized entities (dicts for json and tuples for csv)
    """
    if mimetype == "application/json":
        result = file_.json()
        if isinstance(result, list):
            yield from iter(result)
            return
        else:
            yield result
            return
    elif mimetype == "application/X-lines+json":
        for line in file_.iter_lines(decode_unicode=True):
            yield loads(line)
    elif mimetype == "text/csv":
        csv_reader = reader(file_.iter_lines(decode_unicode=True), csv_dialect)
        header: Sequence[str] = next(csv_reader)
        if process_csv_header:
            header = tuple(process_csv_header(header))
        if tuple_ is None:
            EntityType: Type[NamedTuple] = get_entity_tuple_class(header)
            tuple_ = lambda s: EntityType(*s)  # type: ignore
            assert tuple_ is not None

        yield from (tuple_(row) for row in csv_reader if row)
    else:
        raise ValueError(f"Loading entities from {mimetype} files is not implemented!")


def save_entities(
    entities: Iterable[Union[Dict[str, Any], NamedTuple]],
    file_: TextIO,
    mimetype: str,
    attributes: Optional[Sequence[str]] = None,
    csv_dialect: str = "default",
    tuple_: Optional[Callable[..., Tuple]] = None,
):
    """Write entities to a file.

    CSV files require an attribute order that can be specified by the ``attributes`` parameter.
    The first and second attribute should be ``"ID"`` and ``"href"``.
    The function :py:func:`~qhana_plugin_runner.plugin_utils.entity_marshalling.entity_attribute_sort_key`
    can be used to achieve that order.

    Args:
        entities (Iterable[Union[Dict[str, Any], NamedTuple]]): an iterable of entities as returned by :py:func:`~qhana_plugin_runner.plugin_utils.entity_marshalling.load_entities`
        file_ (TextIO): the file to write the entities into
        mimetype (str): the mime type to use for serialization (supported mimetypes: "application/json", "application/X-lines+json" and "text/csv")
        attributes (Optional[Sequence[str]], optional): A list of attributes in the order they should appear in the csv file. MUST be valid python identifiers! All entities must have all attributes specified here! Defaults to None.
        csv_dialect (str, optional): the csv dialect to use. Defaults to "default".
        tuple_ (Optional[Type[NamedTuple]], optional): the namedtuple class to use (only used with csv mimetype, passed to ``ensure_tuple``). Defaults to None.

    Raises:
        ValueError: if ``mimetype=="text/csv"`` and ``attributes is None``
        ValueError: For unknown mimetypes
    """
    if mimetype == "application/json":
        dump(list(ensure_dict(entities)), file_, separators=(",", ":"))
        file_.write("\n")
    elif mimetype == "application/X-lines+json":
        for entity in ensure_dict(entities):
            file_.write(f"{dumps(entity)}\n")
    elif mimetype == "text/csv":
        if attributes is None:
            raise ValueError(
                "To write entities into the csv format the order of columns must be specified with the attributes parameter!"
            )
        csv_writer = writer(file_, dialect=csv_dialect)
        csv_writer.writerow(attributes)
        if tuple_ is None:
            tuple_ = namedtuple("Entites", attributes)
        csv_writer.writerows(ensure_tuple(entities, tuple_=tuple_))
    else:
        raise ValueError(f"Saving entities to {mimetype} files is not implemented!")
