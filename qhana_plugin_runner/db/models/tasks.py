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
from typing import List, Optional, Sequence, Union

from sqlalchemy.orm import relation, relationship
from sqlalchemy.orm.collections import attribute_mapped_collection
from sqlalchemy.sql import sqltypes as sql
from sqlalchemy.sql.expression import select
from sqlalchemy.sql.schema import (
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
)
from sqlalchemy.ext.associationproxy import association_proxy

from ..db import DB, REGISTRY


@REGISTRY.mapped
@dataclass
class TaskData:
    """Dataclass for key-value store of :class:`ProcessingTask`

    Attributes:
        id (int, optional): automatically generated database id. Use the id to fetch this information from the database.
        key (str, optional): a key in dict
        value (str, optional): a corresponding value in dict
    """

    __tablename__ = "TaskData"

    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(
        init=False,
        metadata={"sa": Column(ForeignKey("ProcessingTask.id"), primary_key=True)},
    )
    key: str = field(metadata={"sa": Column(sql.String(100), primary_key=True)})
    value: Optional[str] = field(metadata={"sa": Column(sql.Text(1000), nullable=True)})


@REGISTRY.mapped
@dataclass
class ProcessingTask:
    """Dataclass for persisting (logical) task information.

    Implements dict-like functionality. Key-value pairs can be specified and accessed as in a dict.

    Attributes:
        id (int, optional): automatically generated database id. Use the id to fetch this information from the database.
        task_name (str): the name of the (logical) task corresponding to this information object
        task_id (Optional[str], optional): the final celery task id to wait for to declare this task finished. If not supplied this task will never be marked as finished.
        started_at (datetime, optional): the moment the task was scheduled. (default :py:func:`~datetime.datetime.utcnow`)
        finished_at (Optional[datetime], optional): the moment the task finished successfully or with an error.
        parameters (str): the parameters for the task. Task parameters should already be prepared and error checked before starting the task.
        data (Optional[dict]): key-value store for additional lightweight task data
        finished_status (Optional[str], optional): the status string with witch the celery task with the ``task_id`` finished. If set then ``task_id`` may not be checked.
        task_log (Optional[str], optional): the task log, task metadata or the error of the finished task. All data results should be file outputs of the task!
        outputs (List[TaskFile], optional): the output data (files) of the task
    """

    __tablename__ = "ProcessingTask"

    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={"sa": Column(sql.INTEGER(), primary_key=True)})
    task_name: str = field(metadata={"sa": Column(sql.String(500))})
    task_id: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.String(64), index=True, nullable=True)}
    )

    started_at: datetime = field(
        default=datetime.utcnow(), metadata={"sa": Column(sql.TIMESTAMP(timezone=True))}
    )
    finished_at: Optional[datetime] = field(
        default=None, metadata={"sa": Column(sql.TIMESTAMP(timezone=True), nullable=True)}
    )

    parameters: str = field(default="", metadata={"sa": Column(sql.Text())})

    _data: dict = field(
        default_factory=dict,
        metadata={
            "sa": relationship(
                "TaskData",
                collection_class=attribute_mapped_collection("key"),
                cascade="all, delete-orphan",
            )
        },
    )

    data = association_proxy(
        "_data", "value", creator=lambda key, value: TaskData(key=key, value=value)
    )

    ui_base_endpoint_url: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.String(200))}
    )

    ui_endpoint_url: Optional[str] = field(  # TODO: maybe rename
        default=None, metadata={"sa": Column(sql.String(200))}
    )

    finished_status: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.String(100))}
    )

    task_log: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.Text(), nullable=True)}
    )

    outputs: List["TaskFile"] = field(
        default_factory=list,
        metadata={"sa": relationship("TaskFile", back_populates="task", lazy="select")},
    )

    @property
    def is_finished(self) -> bool:
        """Return true if the task has finished either successfully or with an error."""
        return self.finished_at is not None

    @property
    def is_ok(self) -> bool:
        """Return true if the task has finished successfully."""
        return self.finished_status == "SUCCESS"

    @property
    def status(self) -> str:
        """Return the finished status of the task.

        If the task is finished but no finished_status was set returns ``"UNKNOWN"``.

        If the task is not finished returns ``"PENDING"``.

        Returns:
            str: ``self.finished_status`` | ``"UNKNOWN"`` | ``"PENDING"``
        """
        if self.is_finished:
            if self.finished_status:
                return self.finished_status
            else:
                return "UNKNOWN"
        return "PENDING"

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


@REGISTRY.mapped
@dataclass
class TaskFile:
    __tablename__ = "TaskFile"

    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(init=False, metadata={"sa": Column(sql.INTEGER(), primary_key=True)})
    task: ProcessingTask = field(
        metadata={
            "sa": relationship(
                "ProcessingTask", back_populates="outputs", lazy="selectin"
            )
        }
    )
    security_tag: str = field(metadata={"sa": Column(sql.String(64), nullable=False)})
    storage_provider: str = field(metadata={"sa": Column(sql.String(64), nullable=False)})
    file_name: str = field(
        metadata={"sa": Column(sql.String(500), index=True, nullable=False)}
    )
    file_storage_data: str = field(metadata={"sa": Column(sql.Text(), nullable=False)})
    file_type: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.String(255), nullable=True)}
    )
    mimetype: Optional[str] = field(
        default=None, metadata={"sa": Column(sql.String(255), nullable=True)}
    )
    created_at: datetime = field(
        default=datetime.utcnow(), metadata={"sa": Column(sql.TIMESTAMP(timezone=True))}
    )
    task_id: Optional[int] = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
        metadata={"sa": Column(sql.INTEGER(), ForeignKey(ProcessingTask.id))},
    )

    def save(self, commit: bool = False):
        """Add this object to the current session and optionally commit the session to persist all objects in the session."""
        DB.session.add(self)
        if commit:
            DB.session.commit()

    @classmethod
    def get_by_id(cls, id_: int) -> Optional["TaskFile"]:
        """Get the object instance by the object id from the database. (None if not found)"""
        return DB.session.execute(select(cls).filter_by(id=id_)).scalar_one_or_none()

    @classmethod
    def get_task_result_files(
        cls, task: Union[int, ProcessingTask]
    ) -> Sequence["TaskFile"]:
        """Get a sequence of task result files (e.g. all files with a file-type that is not "temp-file")."""
        filter_ = (
            cls.file_type != "temp-file",
            cls.task == task if isinstance(task, ProcessingTask) else cls.task_id == task,
        )
        return DB.session.execute(select(cls).filter(*filter_)).scalars().all()
