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

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, cast

from sqlalchemy.sql import sqltypes as sql
from sqlalchemy.sql.expression import ColumnElement, Delete, Select, delete, select
from sqlalchemy.sql.schema import Column

from .mutable_json import MutableJSON
from ..db import DB, REGISTRY
from ...util.plugins import plugin_identifier


@REGISTRY.mapped
@dataclass
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

    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(
        init=False,
        metadata={"sa": Column(sql.INTEGER(), primary_key=True)},
    )
    parent_id: str = field(metadata={"sa": Column(sql.String(550))})
    name: str = field(metadata={"sa": Column(sql.String(500))})
    version: str = field(metadata={"sa": Column(sql.String(50))})
    description: str = field(metadata={"sa": Column(sql.Text())})
    tags: str = field(metadata={"sa": Column(sql.Text())})
    href: str = field(metadata={"sa": Column(sql.Text())})

    @property
    def tag_list(self):
        return [t for t in self.tags.splitlines() if t]

    @property
    def identifier(self):
        return plugin_identifier(self.name, self.version)

    @classmethod
    def get_all(
        cls, for_parents: Optional[Iterable[str]] = None
    ) -> "List[VirtualPlugin]":
        q: Select = select(cls)
        if for_parents is not None:
            q = q.filter(cast(ColumnElement, cls.parent_id).in_(for_parents))
        return DB.session.execute(q).scalars().all()

    @classmethod
    def get_all_by_name(cls, name: str) -> "List[VirtualPlugin]":
        q: Select = select(cls)
        q = q.filter(cls.name == name)
        return DB.session.execute(q).scalars().all()

    @classmethod
    def delete_by_name_and_version(cls, name: str, version: str, commit: bool = False):
        q: Delete = delete(cls)
        q = q.filter(cls.name == name, cls.version == version)
        DB.session.execute(q)
        if commit:
            DB.session.commit()
