# Copyright 2022 QHAna plugin runner contributors.
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

from typing import Any, Iterable, List, Optional, Sequence, cast

from flask import current_app
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import sqltypes as sql
from sqlalchemy.sql.expression import (
    ColumnElement,
    ColumnOperators,
    Delete,
    Select,
    delete,
    literal,
    select,
)

from ...util.plugins import plugin_identifier
from ..db import DB, DB_SIGNALS, REGISTRY
from .mutable_json import JSON_LIKE, MutableJSON

VIRTUAL_PLUGIN_CREATED = DB_SIGNALS.signal("virtual-plugin_created")
VIRTUAL_PLUGIN_REMOVED = DB_SIGNALS.signal("virtual-plugin_removed")


@REGISTRY.mapped_as_dataclass
class VirtualPlugin:
    """A table to keep track of virtual plugins.

    Attributes:
        id (int, optional): database ID of the virtual plugine.
        parent_id (str): the plugin identifier (with or without version) of the plugin that registered this virtual plugin.
        name (str): the name of the virtual plugin.
        version (str): the version of the virtual plugin.
        description (str): a description of the plugin.
        tags (str): a list of tags of the plugin (tags must be separated by newlines).
        href (str): The URL of the REST entry point resource.
    """

    __tablename__ = "VirtualPlugin"

    id: Mapped[int] = mapped_column(sql.INTEGER(), primary_key=True, init=False)
    parent_id: Mapped[str] = mapped_column(sql.String(550))
    name: Mapped[str] = mapped_column(sql.String(500))
    version: Mapped[str] = mapped_column(sql.String(50))
    description: Mapped[str] = mapped_column(sql.Text())
    tags: Mapped[str] = mapped_column(sql.Text())
    href: Mapped[str] = mapped_column(sql.Text())

    @property
    def tag_list(self):
        return [t for t in self.tags.splitlines() if t]

    @property
    def identifier(self):
        return plugin_identifier(self.name, self.version)

    @classmethod
    def get_all(
        cls,
        for_parents: Optional[Iterable[str]] = None,
        filters: Sequence[ColumnOperators] = tuple(),
    ) -> "List[VirtualPlugin]":
        q: Select = select(cls)
        if for_parents is not None:
            q = q.filter(cast(ColumnElement, cls.parent_id).in_(for_parents))
        if filters:
            q = q.filter(*filters)
        return DB.session.execute(q).scalars().all()

    @classmethod
    def get_by_href(
        cls, href: str, parent_id: Optional[str] = None
    ) -> Optional["VirtualPlugin"]:
        q: Select = select(cls)
        q = q.filter(cls.href == href)
        if parent_id is not None:
            q = q.filter(cls.parent_id == parent_id)
        return DB.session.execute(q).scalar_one_or_none()

    @classmethod
    def get_all_by_name(cls, name: str) -> "List[VirtualPlugin]":
        q: Select = select(cls)
        q = q.filter(cls.name == name)
        return DB.session.execute(q).scalars().all()

    @classmethod
    def delete_by_href(
        cls, href: str, parent_id: Optional[str] = None, commit: bool = False
    ):
        q: Delete = delete(cls)
        q = q.filter(cls.href == href)
        if parent_id is not None:
            q = q.filter(cls.parent_id == parent_id)
        DB.session.execute(q)
        if commit:
            DB.session.commit()

    @classmethod
    def delete_by_name_and_version(cls, name: str, version: str, commit: bool = False):
        q: Delete = delete(cls)
        q = q.filter(cls.name == name, cls.version == version)
        DB.session.execute(q)
        if commit:
            DB.session.commit()

    @classmethod
    def exists(cls, query_filter: Sequence[Any] = tuple()) -> bool:
        exists_q = select(literal(True)).select_from(cls).filter(*query_filter).exists()
        return DB.session.execute(select(literal(True)).where(exists_q)).scalar()


@REGISTRY.mapped_as_dataclass
class PluginState:
    """A table to store persistent plugin state.

    The table offers a lightweight key value store for plugins for smallish values.
    Values are automatically serialized and deserialized.
    This table is for storing state that is not related to a specific task.
    Do not use this table to store task state!

    Attributes:
        plugin_id (str): the plugin identifier (with or without version) of the plugin that registered this state.
        key (str): the key under which the state was registered.
        value: (JSON_LIKE): the stored state.
    """

    __tablename__ = "PluginState"

    plugin_id: Mapped[str] = mapped_column(sql.String(550), primary_key=True)
    key: Mapped[str] = mapped_column(sql.String(500), primary_key=True)
    value: Mapped[JSON_LIKE] = mapped_column(MutableJSON)

    @classmethod
    def get_item(cls, plugin_id: str, key: str) -> Optional["PluginState"]:
        """Get a full item record for a given key.

        Args:
            plugin_id (str): the plugin requesting the value (is matched exactly to plugin_id)
            key (str): the key of the state to search for

        Returns:
            Optional[PluginState]: the plugin state record
        """
        q: Select = select(cls)
        q = q.filter(cls.plugin_id == plugin_id, cls.key == key)
        result: Optional[PluginState] = DB.session.execute(q).scalar_one_or_none()
        return result

    @classmethod
    def get_all_items(cls, plugin_id: str) -> Sequence["PluginState"]:
        """Get all plugin state records for a given plugin.

        Args:
            plugin_id (str): the plugin requesting the value (is matched exactly to plugin_id)

        Returns:
            Optional[PluginState]: the plugin state record
        """
        q: Select = select(cls)
        q = q.filter(cls.plugin_id == plugin_id)
        result: Sequence[PluginState] = DB.session.execute(q).scalars().all()
        return result

    @classmethod
    def get_all_items_like(cls, plugin_id: str) -> Sequence["PluginState"]:
        """Get all plugin state records for a given plugin.

        Like get_all_items, but does not match the entire plugin id.
        This uses the database like feature so plugin_ids must not contain any % signs!

        Args:
            plugin_id (str): the plugin requesting the value. A plugin id without version will match any version

        Returns:
            Optional[PluginState]: the plugin state record
        """
        q: Select = select(cls)
        q = q.filter(cast(ColumnElement, cls.plugin_id).like(f"%{plugin_id}%"))
        result: Sequence[PluginState] = DB.session.execute(q).scalars().all()
        return result

    @classmethod
    def get_value(cls, plugin_id: str, key: str, default: Any = ...) -> JSON_LIKE:
        """Get a value for a given key.

        Args:
            plugin_id (str): the plugin requesting the value (is matched exactly to plugin_id)
            key (str): the key of the state to search for
            default (Any, optional): a default value to return if the key was not present.

        Raises:
            KeyError: if the key was not found and no default value was provided

        Returns:
            JSON_LIKE: the stored state value
        """
        q: Select = select(cls)
        q = q.filter(cls.plugin_id == plugin_id, cls.key == key)
        result: Optional[PluginState] = DB.session.execute(q).scalar_one_or_none()
        if result:
            return result.value
        if default is not ...:  # use ellipsis as default as None could be user provided
            return default  # user provided a default value

        raise KeyError(f"Could not find the key {key} for the plugin id {plugin_id}!")

    @classmethod
    def set_value(
        cls, plugin_id: str, key: str, value: JSON_LIKE, commit: bool = False
    ) -> JSON_LIKE:
        """Set state for a given key.

        Args:
            plugin_id (str): the plugin to set the state for (with or without version)
            key (str): the key to store state under
            value (JSON_LIKE): the state to persist
            commit (bool, optional): if true the session will be comitted immediately. Defaults to False.

        Returns:
            JSON_LIKE: the old value if any or None
        """
        existing = cls.get_item(plugin_id=plugin_id, key=key)

        if existing:
            old_value = existing.value
            existing.value = value
            DB.session.add(existing)
        else:
            old_value = None
            new_item = PluginState(plugin_id=plugin_id, key=key, value=value)
            DB.session.add(new_item)

        if commit:
            DB.session.commit()

        return old_value

    @classmethod
    def delete_value(cls, plugin_id: str, key: str, commit: bool = False):
        """Delete state for a given key.

        Args:
            plugin_id (str): the plugin to set the state for (with or without version)
            key (str): the key to store state under
            commit (bool, optional): if true the session will be comitted immediately. Defaults to False.
        """
        existing = cls.get_item(plugin_id=plugin_id, key=key)
        if existing:
            DB.session.delete(existing)
        if commit:
            DB.session.commit()


@REGISTRY.mapped_as_dataclass
class DataBlob:
    __tablename__ = "DataBlob"

    plugin_id: Mapped[str] = mapped_column(sql.String(550), primary_key=True)
    key: Mapped[str] = mapped_column(sql.String(500), primary_key=True)
    value: Mapped[bytes] = mapped_column(sql.LargeBinary())

    @classmethod
    def get_item(cls, plugin_id: str, key: str) -> Optional["DataBlob"]:
        """Get a full item record for a given key.

        Args:
            plugin_id (str): the plugin requesting the value (is matched exactly to plugin_id)
            key (str): the key of the state to search for

        Returns:
            Optional[DataBlob]: the plugin state record
        """
        q: Select = select(cls)
        q = q.filter(cls.plugin_id == plugin_id, cls.key == key)
        result: Optional[DataBlob] = DB.session.execute(q).scalar_one_or_none()
        return result

    @classmethod
    def get_value(cls, plugin_id: str, key: str, default: Any = ...) -> bytes:
        """Get a value for a given key.

        Args:
            plugin_id (str): the plugin requesting the value (is matched exactly to plugin_id)
            key (str): the key of the state to search for
            default (Any, optional): a default value to return if the key was not present.

        Raises:
            KeyError: if the key was not found and no default value was provided

        Returns:
            bytes: the stored state value, or the default value
        """
        q: Select = select(cls)
        q = q.filter(cls.plugin_id == plugin_id, cls.key == key)
        result: Optional[DataBlob] = DB.session.execute(q).scalar_one_or_none()
        if result:
            return result.value
        if default is not ...:  # use ellipsis as default as None could be user provided
            return default  # user provided a default value

        raise KeyError(f"Could not find the key {key} for the plugin id {plugin_id}!")

    @classmethod
    def set_value(
        cls, plugin_id: str, key: str, value: bytes, commit: bool = False
    ) -> Optional[bytes]:
        """Set state for a given key.

        Args:
            plugin_id (str): the plugin to set the state for (with or without version)
            key (str): the key to store state under
            value (bytes): the state to persist
            commit (bool, optional): if true the session will be comitted immediately. Defaults to False.

        Returns:
            bytes: the old value if any or None
        """
        existing = cls.get_item(plugin_id, key)

        if existing:
            old_value = existing.value
            existing.value = value
            DB.session.add(existing)
        else:
            old_value = None
            new_item = DataBlob(plugin_id=plugin_id, key=key, value=value)
            DB.session.add(new_item)

        if commit:
            DB.session.commit()

        return old_value

    @classmethod
    def delete_value(cls, plugin_id: str, key: str, commit: bool = False):
        """Delete state for a given key.

        Args:
            plugin_id (str): the plugin to set the state for (with or without version)
            key (str): the key to store state under
            commit (bool, optional): if true the session will be comitted immediately. Defaults to False.
        """
        existing = cls.get_item(plugin_id=plugin_id, key=key)
        if existing:
            DB.session.delete(existing)
        if commit:
            DB.session.commit()
