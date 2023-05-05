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

from datetime import datetime
from typing import List, Optional, Sequence, Union

from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.orderinglist import OrderingList, ordering_list
from sqlalchemy.sql import sqltypes as sql
from sqlalchemy.sql.expression import select
from sqlalchemy.sql.schema import ForeignKey

from .mutable_json import MutableJSON
from ..db import DB, REGISTRY


@REGISTRY.mapped_as_dataclass
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

    id: Mapped[int] = mapped_column(ForeignKey("ProcessingTask.id"), primary_key=True)
    step_id: Mapped[str] = mapped_column(sql.String(500))
    number: Mapped[int] = mapped_column(sql.Integer(), init=False, primary_key=True)
    href: Mapped[str] = mapped_column(sql.String(500))
    ui_href: Mapped[str] = mapped_column(sql.String(500))
    cleared: Mapped[bool] = mapped_column(sql.Boolean(), default=False)


@REGISTRY.mapped_as_dataclass
class ProcessingTask:
    """Dataclass for persisting (logical) task information.

    Attributes:
        id (int, optional): automatically generated database id. Use the id to fetch this information from the database.
        task_name (str): the name of the (logical) task corresponding to this information object
        started_at (datetime, optional): the moment the task was scheduled. (default :py:func:`~datetime.datetime.utcnow`)
        finished_at (Optional[datetime], optional): the moment the task finished successfully or with an error.
        parameters (str): the parameters for the task. Task parameters should already be prepared and error checked before starting the task.
        data (Union[dict, list, str, float, int, bool, None]): mutable JSON-like store for additional lightweight task data. Default value is empty dict.
        steps (OrderingList[Step]): ordered list of steps of type :class:`Step`. Index ``number`` automatically increases when new elements are appended. Note: only use :meth:`add_next_step` to add a new step. Steps must not be deleted.
        current_step (int): index of last added step.
        progress_value (float): current progress value. ``None`` by default.
        progress_start (float): progress start value.
        progress_target (float): progress target value.
        progress_unit (str): progress unit (default: "%").
        task_status (Optional[str], optional): the status string of the plugin execution, can only be ``PENDING``, ``SUCCESS``, or ``ERROR``.
        task_log (str): the task log, task metadata or the error of the finished task. All data results should be file outputs of the task!
        outputs (List[TaskFile], optional): the output data (files) of the task
    """

    __tablename__ = "ProcessingTask"

    id: Mapped[int] = mapped_column(sql.INTEGER(), init=False, primary_key=True)
    task_name: Mapped[str] = mapped_column(sql.String(500))

    started_at: Mapped[datetime] = mapped_column(
        sql.TIMESTAMP(timezone=True), default=datetime.utcnow()
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        sql.TIMESTAMP(timezone=True), default=None, nullable=True
    )

    parameters: Mapped[str] = mapped_column(sql.Text(), default="")

    data: Mapped[Union[dict, list, str, float, int, bool, None]] = mapped_column(
        MutableJSON, default_factory=dict
    )

    multi_step: Mapped[bool] = mapped_column(sql.Boolean(), default=False)

    steps: Mapped[OrderingList] = relationship(
        "Step",
        order_by="Step.number",
        collection_class=ordering_list("number"),
        cascade="all, delete-orphan",
        default_factory=list,
    )

    current_step: Mapped[int] = mapped_column(sql.Integer(), default=-1)

    progress_value: Mapped[float] = mapped_column(
        sql.Float(), default=None, nullable=True
    )
    progress_start: Mapped[float] = mapped_column(sql.Float(), default=0)
    progress_target: Mapped[float] = mapped_column(sql.Float(), default=100)
    progress_unit: Mapped[str] = mapped_column(sql.String(100), default="%")

    task_status: Mapped[Optional[str]] = mapped_column(sql.String(100), default=None)

    task_log: Mapped[str] = mapped_column(sql.Text(), default="", nullable=False)

    outputs: Mapped[List["TaskFile"]] = relationship(
        "TaskFile", back_populates="task", lazy="select", default_factory=list
    )

    @property
    def is_finished(self) -> bool:
        """Return true if the task has finished either successfully or with an error."""
        return self.finished_at is not None

    @property
    def is_ok(self) -> bool:
        """Return true if the task has finished successfully."""
        return self.task_status == "SUCCESS"

    @property
    def status(self) -> str:
        """Return the finished status of the task.

        If the task is finished but no task_status was set returns ``"UNKNOWN"``.

        If the task is not finished returns ``"PENDING"``.

        Returns:
            str: ``self.task_status`` | ``"UNKNOWN"`` | ``"PENDING"``
        """
        if self.is_finished:
            if self.task_status:
                return self.task_status
            else:
                return "UNKNOWN"
        return "PENDING"

    def clear_previous_step(self, commit: bool = False):
        """Set ``"cleared"`` of previous step to ``true`` if available. Note: call before calling add_next_step."""
        try:
            step: Step = self.steps[self.current_step]
            step.cleared = True
            DB.session.add(step)
            if commit:
                DB.session.commit()
        except:
            pass

    def add_next_step(self, href: str, ui_href: str, step_id: str, commit: bool = False):
        """Adds new step for multi-step plugin.

        Args:
            href (str): The URL of the REST entry point resource.
            ui_href (str): The URL of the micro frontend that corresponds to the REST entry point resource.
            step_id (str): ID of step, e.g., ``"step1"`` or ``"step2b"``, is automatically appended to previous step

        Raises:
            AssertionError: raised in case the previous step was not cleared before this method is called.
        """
        if self.current_step >= 0:
            if not self.steps[self.current_step].cleared:
                raise AssertionError(
                    "Previous step must be cleared first before adding a new step!"
                )
            step_id = step_id
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

    def add_task_log_entry(self, task_log: str, commit: bool = False):
        """Appends ``task_log`` separated by a new line.

        Args:
            task_log (str): new entry to be added
        """
        if self.task_log:
            self.task_log += "\n" + task_log
        else:
            self.task_log = task_log

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


@REGISTRY.mapped_as_dataclass
class TaskFile:
    __tablename__ = "TaskFile"

    id: Mapped[int] = mapped_column(sql.INTEGER(), primary_key=True, init=False)
    task: Mapped[ProcessingTask] = relationship(
        "ProcessingTask", back_populates="outputs", lazy="selectin"
    )
    security_tag: Mapped[str] = mapped_column(sql.String(64), nullable=False)
    storage_provider: Mapped[str] = mapped_column(sql.String(64), nullable=False)
    file_name: Mapped[str] = mapped_column(sql.String(500), index=True, nullable=False)
    file_storage_data: Mapped[str] = mapped_column(sql.Text(), nullable=False)
    file_type: Mapped[Optional[str]] = mapped_column(sql.String(255), nullable=True)
    mimetype: Mapped[Optional[str]] = mapped_column(
        sql.String(255), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        sql.TIMESTAMP(timezone=True), default=datetime.utcnow()
    )
    task_id: Mapped[Optional[int]] = mapped_column(
        sql.INTEGER(),
        ForeignKey(ProcessingTask.id),
        default=None,
        init=False,
        repr=False,
        compare=False,
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
