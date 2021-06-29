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

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.sql import sqltypes as sql
from sqlalchemy.sql.expression import select
from sqlalchemy.sql.schema import Column

from ..db import DB, REGISTRY


@REGISTRY.mapped
@dataclass
class ProcessingTask:
    __tablename__ = "ProcessingTask"

    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={"sa": Column(sql.INTEGER(), primary_key=True)})
    task_name: str = field(metadata={"sa": Column(sql.String(500))})
    task_id: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.String(32), index=True, nullable=True)}
    )

    started_at: datetime = field(
        default=datetime.utcnow(), metadata={"sa": Column(sql.TIMESTAMP(timezone=True))}
    )
    finished_at: Optional[datetime] = field(
        default=None, metadata={"sa": Column(sql.TIMESTAMP(timezone=True), nullable=True)}
    )

    parameters: str = field(default="", metadata={"sa": Column(sql.Text())})
    input_files: List[Dict[str, str]] = field(
        default_factory=list, metadata={"sa": Column(sql.JSON())}
    )

    finished_status: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.String(100))}
    )

    task_result: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.Text(), nullable=True)}
    )
    output_files: Optional[List[Dict[str, str]]] = field(
        default=None, metadata={"sa": Column(sql.JSON(), nullable=True)}
    )

    def save(self, commit: bool = False):
        """Add this object to the current session and optionally commit the session to persist all objects in the session."""
        DB.session.add(self)
        if commit:
            DB.session.commit()

    @classmethod
    def get_by_id(cls, id_: int) -> Optional["ProcessingTask"]:
        """Get the object instance by the object id from the database. (None if not found)"""
        return DB.session.execute(select(cls).filter_by(id=id_)).scalar_one_or_none()

    @classmethod
    def get_by_task_id(cls, task_id) -> Optional["ProcessingTask"]:
        """Get the object instance by the task_id from the database. (None if not found)"""
        return DB.session.execute(
            select(cls).filter_by(task_id=task_id)
        ).scalar_one_or_none()
