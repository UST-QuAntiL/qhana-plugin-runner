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
from sqlalchemy.ext.orderinglist import OrderingList, ordering_list
from sqlalchemy.sql import sqltypes as sql
from sqlalchemy.sql.expression import select
from sqlalchemy.sql.schema import (
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    PrimaryKeyConstraint,
)
from sqlalchemy.ext.associationproxy import AssociationProxy, association_proxy

from ..db import DB, REGISTRY


@REGISTRY.mapped
@dataclass
class Step:
    """Step for multi-step plugins

    Attributes:
        id (int, optional): ID of corresponding :class:`ProcessingTask` entry. Use the id to fetch this information from the database.
        step_id (str): ID of step, e.g., ``"step1"`` or ``"step1.step2b"``.
        href (str): The URL of the REST entry point resource.
        ui_href (str): The URL of the micro frontend that corresponds to the REST entry point resource.
        cleared (bool): ``false`` if step is awaiting input, only last step in list can be marked as ``false``.
    """

    __tablename__ = "Step"

    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(
        metadata={"sa": Column(ForeignKey("ProcessingTask.id"), primary_key=True)},
    )
    step_id: str = field(metadata={"sa": Column(sql.String(500))})
    number: int = field(
        init=False, metadata={"sa": Column(sql.Integer(), primary_key=True)}
    )
    href: str = field(metadata={"sa": Column(sql.String(200))})
    ui_href: str = field(metadata={"sa": Column(sql.String(200))})
    cleared: bool = field(metadata={"sa": Column(sql.Boolean())}, default=False)


@REGISTRY.mapped
@dataclass
class TaskData:
    """Dataclass for key-value store of :class:`ProcessingTask`

    Attributes:
        id (int): ID of corresponding :class:`ProcessingTask` entry. Use the id to fetch this information from the database.
        key (str): a key in dict
        value (str, optional): a corresponding value in dict
    """

    __tablename__ = "TaskData"

    __sa_dataclass_metadata_key__ = "sa"

    id: int = field(
        metadata={
            "sa": Column(
                ForeignKey("ProcessingTask.id"), primary_key=True, nullable=False
            )
        },
    )
    key: str = field(metadata={"sa": Column(sql.String(100), primary_key=True)})
    value: Optional[str] = field(metadata={"sa": Column(sql.Text(1000), nullable=True)})


@REGISTRY.mapped
@dataclass
class ProcessingTask:
    """Dataclass for persisting (logical) task information.

    Attributes:
        id (int, optional): automatically generated database id. Use the id to fetch this information from the database.
        task_name (str): the name of the (logical) task corresponding to this information object
        task_id (Optional[str], optional): the final celery task id to wait for to declare this task finished. If not supplied this task will never be marked as finished. In multi-step plugins, this attribute is also used for intermediate task results. Only needed to check task status of :py:func:`qhana-plugin-runner.qhana_plugin_runner.tasks.save_task_result`.
        started_at (datetime, optional): the moment the task was scheduled. (default :py:func:`~datetime.datetime.utcnow`)
        finished_at (Optional[datetime], optional): the moment the task finished successfully or with an error.
        parameters (str): the parameters for the task. Task parameters should already be prepared and error checked before starting the task.
        data (dict): dict-like key-value store for additional lightweight task data. New elements of type :class:`TaskData` can be added or retrieved as in a dict using ``key`` as key.
        multi_step (bool): set to ``True`` if task data is used for a multi-step plugin.
        steps (OrderingList[Step]): ordered list of steps of type :class:`Step`. Index ``number`` automatically increases when new elements are appended. Note: only use :meth:`add_next_step` to add a new step. Steps must not be deleted.
        current_step (int): index of last added step.
        progress_value (int): progress value in multi-step plugins.
        progress_start (int): progress start value in multi-step plugins.
        progress_target (int): progress target value in multi-step plugins.
        progress_unit (str): progress unit in multi-step plugins (default: "%").
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
        "_data", "value", creator=lambda key, value: TaskData(id=id, key=key, value=value)
    )

    multi_step: bool = field(default=False, metadata={"sa": Column(sql.Boolean())})

    steps: OrderingList = field(
        default_factory=list,
        metadata={
            "sa": relationship(
                "Step",
                order_by="Step.number",
                collection_class=ordering_list("number"),
                cascade="all, delete-orphan",
            )
        },
    )

    current_step: int = field(default=-1, metadata={"sa": Column(sql.Integer())})

    progress_value: int = field(default=0, metadata={"sa": Column(sql.Integer())})
    progress_start: int = field(default=0, metadata={"sa": Column(sql.Integer())})
    progress_target: int = field(default=100, metadata={"sa": Column(sql.Integer())})
    progress_unit: str = field(default="%", metadata={"sa": Column(sql.String(20))})

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

    def clear_previous_step(self, commit: bool = False):
        """Set ``"cleared"`` of previous step to ``true``. Note: call before calling add_next_step."""
        step: Step = self.steps[self.current_step]
        step.cleared = True
        DB.session.add(step)
        if commit:
            DB.session.commit()

    def add_next_step(self, href: str, ui_href: str, step_id: str, commit: bool = False):
        """Adds new step for multi-step plugin.

        Args:
            href (str): The URL of the REST entry point resource.
            ui_href (str): The URL of the micro frontend that corresponds to the REST entry point resource.
            step_id (str): ID of step, e.g., ``"step1"`` or ``"step2b"``, is automatically appended to previous step
        """
        if self.current_step >= 0:
            if not self.steps[self.current_step].cleared:
                raise ValueError(
                    "Previous step must be cleared first before adding a new step!"
                )
            step_id = self.steps[self.current_step].step_id + "." + step_id
        else:
            self.multi_step = True

        self.current_step += 1
        new_step: Step = Step(
            id=self.id,
            step_id=step_id,
            href=href,
            ui_href=ui_href,
        )
        self.steps.append(new_step)

        DB.session.add(new_step)
        DB.session.add(self)
        if commit:
            DB.session.commit()

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
