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

"""Module containing celery tasks."""

from datetime import datetime
from typing import Optional

from blinker import Namespace
from celery.result import AsyncResult
from celery.utils.log import get_task_logger
from flask.globals import current_app

from qhana_plugin_runner.db.models.tasks import ProcessingTask

# make sure that celery tasks in plugin utils are always loaded
from qhana_plugin_runner.plugin_utils import interop  # noqa

from .celery import CELERY

_name = "qhana-plugin-runner"

TASK_LOGGER = get_task_logger(_name)

TASK_SIGNALS = Namespace()

TASK_STATUS_CHANGED = TASK_SIGNALS.signal("task-status-changed")
"""Signal for changes of the task status."""

TASK_STEPS_CHANGED = TASK_SIGNALS.signal("task-steps-changed")
"""Signal for changes of the step lists of tasks. Changes include new steps and clearing existing steps."""

TASK_DETAILS_CHANGED = TASK_SIGNALS.signal("task-details-changed")
"""Signal for any changes of a task excluding status and step events."""


# TODO add periodic cleanup task to remove old results from the database!
@CELERY.task(name=f"{_name}.add-step", bind=True, ignore_result=True)
def add_step(
    self,
    task_log: str,
    db_id: int,
    step_id: str,
    href: str,
    ui_href: str,
    prog_value: Optional[int] = None,
    prog_start: int = 0,
    prog_target: int = 100,
    prog_unit: str = "%",
):
    """Add next step in a multi-step plugin to the database.

    Attributes:
        step_id (str): ID of step, e.g., ``"step1"`` or ``"step1.step2b"``.
        href (str): The *absolute* URL of the REST entry point resource.
        ui_href (str): The *absolute* URL of the micro frontend that corresponds to the REST entry point resource.
        prog_value (int|None): progress value. If None, use <nr of steps>/<nr of steps + 1> as progress.
        prog_start (int): progress start value.
        prog_target (int): progress target value.
        prog_unit (str): progress unit(default: "%").
    """
    if not isinstance(task_log, str):
        raise TypeError(
            f"The task log / task metadata must be of type str to be stored in the database! (expected str but got {type(task_log)})"
        )

    TASK_LOGGER.debug(f"Adding next step with db id '{db_id}'")
    task_data: ProcessingTask = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        # TODO use better fitting error
        raise KeyError(f"Could not find db entry for id {db_id}, add_step failed!")

    if prog_value is None:
        # if no progress is given, set progress as "<steps completed>/<all steps>"
        # without specific unit
        nr_of_steps = len(task_data.steps)
        prog_value = nr_of_steps
        prog_start = 0
        prog_target = nr_of_steps + 1
        prog_unit = ""

    task_data.progress_start = prog_start
    task_data.progress_target = prog_target
    task_data.progress_unit = prog_unit
    task_data.progress_value = prog_value
    task_data.add_next_step(href=href, ui_href=ui_href, step_id=step_id)

    if isinstance(task_log, str):
        task_data.add_task_log_entry(task_log)
    else:
        task_data.add_task_log_entry(repr(task_log))

    task_data.save(commit=True)
    TASK_LOGGER.debug(f"Save task log for task with db id '{db_id}' successful.")

    app = current_app._get_current_object()
    TASK_STEPS_CHANGED.send(app, task_id=db_id)
    TASK_DETAILS_CHANGED.send(app, task_id=db_id)

    AsyncResult(
        self.request.parent_id if self.request.parent_id else self.request.id, app=CELERY
    ).forget()


@CELERY.task(name=f"{_name}.save-result", bind=True, ignore_result=True)
def save_task_result(self, task_log: str, db_id: int):
    """Save the task log in the database and update the final task status of the database task."""
    if task_log is not None and not isinstance(task_log, str):
        raise TypeError(
            f"The task log / task metadata must be of type str to be stored in the database! (expected str but got {type(task_log)})"
        )

    TASK_LOGGER.debug(f"Saving result for task with db id '{db_id}'")
    task_data: ProcessingTask = ProcessingTask.get_by_id(id_=db_id)
    if task_data is None:
        # TODO use better fitting error
        raise KeyError(f"Could not find db entry for id {db_id}, saving task log failed!")

    if task_data.progress_value:
        task_data.progress_value = task_data.progress_target

    task_data.task_status = "SUCCESS"
    had_uncleared_step = task_data.has_uncleared_step
    task_data.clear_previous_step()
    task_data.finished_at = datetime.utcnow()
    if task_log is None:
        pass
    elif isinstance(task_log, str):
        task_data.add_task_log_entry(task_log)
    else:
        task_data.add_task_log_entry(repr(task_log))

    task_data.save(commit=True)
    TASK_LOGGER.debug(f"Save task log for task with db id '{db_id}' successful.")

    app = current_app._get_current_object()
    TASK_STATUS_CHANGED.send(app, task_id=db_id)
    TASK_DETAILS_CHANGED.send(app, task_id=db_id)
    if had_uncleared_step:
        TASK_STEPS_CHANGED.send(app, task_id=db_id)

    # TODO: clean TaskData entries

    AsyncResult(self.request.parent_id, app=CELERY).forget()


@CELERY.task(name=f"{_name}.save-error", bind=True, ignore_result=True)
def save_task_error(self, failing_task_id: str, db_id: int):
    """Save the error as the result of the root task in the database."""
    result = AsyncResult(failing_task_id, app=CELERY)
    exc = result.result
    traceback = result.traceback

    TASK_LOGGER.error(
        f"Sub-Task {failing_task_id} of Task with db id {db_id} raised exception: {exc!r}\n{traceback}"
    )

    task_data: ProcessingTask = ProcessingTask.get_by_id(id_=db_id)

    if task_data is None:
        TASK_LOGGER.error(
            f"Cannot save error for task with db id {db_id}, no db entry found!"
        )
        return  # TODO start new error logging task or save to extra db table

    task_data.task_status = "FAILURE"
    task_data.finished_at = datetime.utcnow()
    task_data.add_task_log_entry(f"{exc!r}\n\n{traceback}")

    task_data.save(commit=True)

    app = current_app._get_current_object()
    TASK_STATUS_CHANGED.send(app, task_id=db_id)
    TASK_DETAILS_CHANGED.send(app, task_id=db_id)

    # TODO: maybe clean TaskData entries

    result.forget()
